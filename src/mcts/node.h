/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h"
#include "utils/mutex.h"

namespace lczero {

// Terminology:
// * Edge    - a potential edge with a move and policy information.
// * Node    - a realized edge with number of visits and evaluation.
// * LowNode - a node with number of visits, evaluation and edges.
//
// Storage:
// * Potential edges are stored in a simple array inside the LowNode as edges_.
// * Realized edges are stored at index they have in edges_ in a logical array
//   stored in LowNode as a single static and several dynamic arrays, allocated
//   on-demand.
// * Realized edges have a copy of their potential edge counterpart, index_
//   among potential edges and are linked to the target LowNode via the
//   low_node_ pointer.
//
// Example:
//                                 LowNode
//                                    |
//        +-------------+-------------+----------------+--------------+
//        |              |            |                |              |
//   Edge 0(Nf3)    Edge 1(Bc5)     Edge 2(a4)     Edge 3(Qxf7)    Edge 4(a3)
//    (dangling)         |           (dangling)        |           (dangling)
//                   Node, Q=0.5                    Node, Q=-0.2
//
//  Is represented as:
// +-----------------+
// | LowNode         |
// +-----------------+                      +--------+
// | edges_          | -------------------> | Edge[] |
// |                 |    +------------+    +--------+
// | children_       | -> | Node[]     |    | Nf3    |
// |                 |    +------------+    | Bc5    |
// | ...             |    | edge_      |    | a4     |
// |                 |    | index_ = 1 |    | Qxf7   |
// |                 |    | wl_ = 0.5  |    | a3     |
// |                 |    +------------+    +--------+
// |                 |    |            |
// |                 |    |            |
// +-----------------+    |            |
//                        +------------+
//                        | edge_      |
//                        | index_ = 3 |
//                        | q_ = -0.2  |
//                        +------------+

// Define __i386__  or __arm__ also for 32 bit Windows.
#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

  static void SortEdges(Edge* edges, int num_edges);

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

struct NNEval {
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges;

  // 4 byte fields.
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;

  // 1 byte fields.
  // Number of edges in @edges.
  uint8_t num_edges = 0;
};

typedef std::pair<GameResult, GameResult> Bounds;

enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase };

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

class LowNode;
class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  // Takes own @index in the parent.
  Node()
      : terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {}
  // Takes own @edge and @index in the parent.
  Node(const Edge& edge, uint16_t index)
      : edge_(edge),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON) {}
  ~Node() { UnsetLowNode(); }

  // Atomics prevent use of the default move constructor version.
  Node(Node&& move_from) { *this = std::move(move_from); }
  // Atomics prevent use of the default move assignment version.
  // Only works in the "constructed" state decided based on index_.
  Node& operator=(Node&& move_from);

  // Completely reset node to "constructed" state.
  void Reset();
  // Trim node, resetting everything except edge and index.
  void Trim();

  // Get first child.
  Node* GetChild() const;

  // Returns whether a node has children.
  bool HasChildren() const;

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const;
  uint32_t GetChildrenVisits() const;
  uint32_t GetTotalVisits() const;
  // Returns n + n_in_flight.
  int GetNStarted() const {
    return n_ + n_in_flight_.load(std::memory_order_acquire);
  }

  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }

  uint8_t GetNumEdges() const;

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 1.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the node not terminal and recomputes bounds, visits and values.
  // Changes low node as well unless @also_low_node is false.
  void MakeNotTerminal(bool also_low_node = true);
  void SetBounds(GameResult lower, GameResult upper);

  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate(int multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=multivisit)
  // * N-in-flight (-=multivisit)
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  // When search decides to treat one visit as several (in case of collisions
  // or visiting terminal nodes several times), it amplifies the visit by
  // incrementing n_in_flight.
  void IncrementNInFlight(int multivisit);

  // Returns range for iterating over edges.
  ConstIterator Edges() const;
  Iterator Edges();

  // Returns range for iterating over child nodes with N > 0.
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();

  // Deletes all children except one.
  // The node provided may be moved, so should not be relied upon to exist
  // afterwards.
  void ReleaseChildrenExceptOne(Node* node_to_save) const;

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const {
    return edge_.GetMove(as_opponent);
  }
  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise or when turning terminal).
  // Must be in [0,1].
  float GetP() const { return edge_.GetP(); }
  void SetP(float val) { edge_.SetP(val); }

  LowNode* GetLowNode() const { return low_node_; }

  void SetLowNode(LowNode* low_node);
  void UnsetLowNode();

  // Debug information about the node.
  std::string DebugString() const;
  // Return string describing the edge from node's parent to its low node in the
  // Graphviz dot format.
  std::string DotEdgeString(bool as_opponent = false,
                            const LowNode* parent = nullptr) const;
  // Return string describing the graph starting at this node in the Graphviz
  // dot format.
  std::string DotGraphString(bool as_opponent = false) const;

  // Returns true if graph under this node has every n_in_flight_ == 0 and
  // prints offending nodes and low nodes and stats to cerr otherwise.
  bool ZeroNInFlight() const;

  void SortEdges() const;

  // Index in parent's edges - useful for correlated ordering.
  uint16_t Index() const { return index_; }

  // Check if node was realized (not just constructed).
  bool Realized() const {
    return index_.load(std::memory_order_acquire) < kMagicIndexAssigned;
  }

 private:
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from
  // the perspective of the player-to-move for the position. WL stands for "W
  // minus L". Is equal to Q if draw score is 0.
  double wl_ = 0.0f;

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Pointer to the low node.
  LowNode* low_node_ = nullptr;

  // 4 byte fields.
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move.
  float d_ = 0.0f;
  // Estimated remaining plies.
  float m_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;
  // (AKA virtual loss.) How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  std::atomic<uint32_t> n_in_flight_ = 0;

  // Move and policy for this edge.
  Edge edge_;

  // 2 byte fields.
  // Magic index constant - Node was constructed.
  constexpr static uint16_t kMagicIndexConstructed = 65535;
  // Magic index constant - Node is being assigned.
  constexpr static uint16_t kMagicIndexAssigned = 32767;
  // Index among parent's edges.
  std::atomic<uint16_t> index_ = kMagicIndexConstructed;

  // 1 byte fields.
  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or
  // draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
};

// Check that Node still fits into an expected cache line size.
static_assert(sizeof(Node) <= 64, "Node is too large");

// Compute running sums for the @vals array. Initialize the sum with @init and
// store results before adding the next value from @vals. Return array with
// results. The result array is one item larger that @vals. (result[0] = init,
// result[1] = init + vals[0])
template <class T, size_t S>
constexpr static std::array<T, S + 1> RunningSumsBefore(
    T init, const std::array<T, S>& vals) {
  std::array<size_t, S + 1> tmp{};
  T sum = init;
  size_t i = 0;
  for (i = 0; i < S; ++i) {
    tmp[i] = sum;
    sum += vals[i];
  }
  tmp[i] = sum;

  return tmp;
}

// Compute running sums for the @vals array. Initialize the sum with @init and
// store results after adding the next value from @vals. Return array with
// results. (result[0] = init + vals[0])
template <class T, size_t S>
constexpr static std::array<T, S> RunningSumsAfter(
    T init, const std::array<T, S>& vals) {
  std::array<size_t, S> tmp{};
  T sum = init;
  for (size_t i = 0; i < S; ++i) {
    sum += vals[i];
    tmp[i] = sum;
  }

  return tmp;
}

class LowNode {
 public:
  LowNode()
      : terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        is_transposition(false) {}
  // Init from from another low node, but use it for NNEval only.
  LowNode(const LowNode& p)
      : wl_(p.wl_),
        d_(p.d_),
        m_(p.m_),
        num_edges_(p.num_edges_),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        is_transposition(false) {
    assert(p.edges_);
    edges_ = std::make_unique<Edge[]>(num_edges_);
    std::memcpy(edges_.get(), p.edges_.get(), num_edges_ * sizeof(Edge));
  }
  // Init @edges_ with moves from @moves and 0 policy.
  LowNode(const MoveList& moves)
      : num_edges_(moves.size()),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        is_transposition(false) {
    edges_ = Edge::FromMovelist(moves);
  }
  // Init @edges_ with moves from @moves and 0 policy.
  // Also create the first child at @index.
  LowNode(const MoveList& moves, uint16_t index)
      : num_edges_(moves.size()),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        is_transposition(false) {
    edges_ = Edge::FromMovelist(moves);
    new (&static_children_[0]) Node(edges_[index], index);
  }

  // Manual memory allocation requires special destructor.
  ~LowNode() { ReleaseChildren(); }

  void SetNNEval(const NNEval* eval) {
    assert(!edges_);
    assert(n_ == 0);

    edges_ = std::make_unique<Edge[]>(eval->num_edges);
    std::memcpy(edges_.get(), eval->edges.get(),
                eval->num_edges * sizeof(Edge));

    wl_ = eval->q;
    d_ = eval->d;
    m_ = eval->m;

    num_edges_ = eval->num_edges;
  }

  // Gets the first realized edge.
  Node* GetChild();

  // Returns whether a node has children.
  bool HasChildren() const { return num_edges_ > 0; }

  uint32_t GetN() const { return n_; }
  uint32_t GetChildrenVisits() const { return n_ - 1; }

  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }

  // Returns whether the node is known to be draw/loss/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  Terminal GetTerminalType() const { return terminal_type_; }

  uint8_t GetNumEdges() const { return num_edges_; }
  // Gets pointer to the start of the edge array.
  Edge* GetEdges() const { return edges_.get(); }

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the low node not terminal and recomputes bounds, visits and values
  // using incoming @node.
  void MakeNotTerminal(const Node* node);
  void SetBounds(GameResult lower, GameResult upper);

  // Decrements n-in-flight back.
  void CancelScoreUpdate(int multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=multivisit)
  // * N-in-flight (-=multivisit)
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, int multivisit);

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  // The child provided will be moved!
  void ReleaseChildrenExceptOne(Node* child_to_save);

  // Return move policy for edge/node at @index.
  const Edge& GetEdgeAt(uint16_t index) const;

  // Debug information about the node.
  std::string DebugString() const;
  // Return string describing this node in the Graphviz dot format.
  std::string DotNodeString() const;

  void SortEdges() {
    assert(edges_);
    assert(n_ == 0);

    Edge::SortEdges(edges_.get(), num_edges_);
  }

  // Add new parent with @n_in_flight visits.
  void AddParent() {
    ++num_parents_;

    assert(num_parents_ > 0);

    is_transposition |= num_parents_ > 1;
  }
  // Remove parent and its first visit.
  void RemoveParent() {
    assert(num_parents_ > 0);
    --num_parents_;
  }
  uint16_t GetNumParents() const { return num_parents_; }
  bool IsTransposition() const { return is_transposition; }

  // Return realized edge at specified index, or nullptr.
  Node* GetChildAt(uint16_t index);
  // Return realized edge at specified index, creating it if necessary.
  // Initializes a new child if @init is true.
  Node* InsertChildAt(uint16_t index, bool init = true);

 private:
  // How many children/realized edges are inlined here.
  constexpr static size_t kStaticChildrenArraySize = 2;
  // Number of dynamically allocated array for children/realized edges.
  constexpr static size_t kDynamicChildrenArrayCount = 1;
  // Sizes of dynamically allocated array for children/realized edges. All
  // arrays have fixed size, except the last one that holds the rest of
  // children/realized edges.
  constexpr static std::array<size_t, kDynamicChildrenArrayCount - 1>
      kDynamicChildrenArraySizes = {};
  // Starts of dynamically allocated array for children/realized edges.
  constexpr static std::array<size_t, kDynamicChildrenArrayCount>
      kDynamicChildrenArrayStarts = RunningSumsBefore(
          kStaticChildrenArraySize, kDynamicChildrenArraySizes);
  // Ends of dynamically allocated array for children/realized edges.
  constexpr static std::array<size_t, kDynamicChildrenArrayCount - 1>
      kDynamicChildrenArrayEnds = RunningSumsAfter(kStaticChildrenArraySize,
                                                   kDynamicChildrenArraySizes);
  constexpr static size_t kDynamicChildrenArrayKnownTotalSize =
      kStaticChildrenArraySize;  // kDynamicChildrenArrayEnds.back();

  // Find a place where an existing child at @index is in child arrays and
  // return it.
  Node* FindPlaceOf(uint16_t index);
  // Allocate a new child array @children of specified @size when
  // @already_allocated children were allocated (passed to avoid another load).
  void Allocate(uint16_t size, uint16_t* already_allocated,
                std::atomic<Node*>* children);

  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // Array of the first few real edges, preallocated here.
  Node static_children_[kStaticChildrenArraySize];

  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from the
  // perspective of the player-to-move for the position.
  // WL stands for "W minus L". Is equal to Q if draw score is 0.
  double wl_ = 0.0f;

  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges_;

  // Arrays with children/real edges with higher indexes, allocated on demand.
  // The last array num_edges_ - children_3_end_
  std::array<std::atomic<Node*>, kDynamicChildrenArrayCount> dynamic_children_ =
      {};

  // 4 byte fields.
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move.
  float d_ = 0.0f;
  // Estimated remaining plies.
  float m_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;

  // 2 byte fields.
  // How many realized children were already allocated.
  std::atomic<uint16_t> allocated_children_ = kStaticChildrenArraySize;
  // Number of parents.
  uint16_t num_parents_ = 0;

  // 1 byte fields.
  // Number of edges in @edges_.
  uint8_t num_edges_ = 0;
  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  // Low node is a transposition (for ever).
  bool is_transposition : 1;
};

// Check important field sizes.
static_assert(sizeof(std::atomic<uint32_t>) <= 4,
              "Unexpected uint32_t is too large.");
static_assert(sizeof(std::atomic<uint16_t>) <= 2,
              "Unexpected uint16_t atomic is too large.");
// Check that LowNode still fits into an expected cache line size.
static_assert(sizeof(LowNode) <= 128, "LowNode is too large.");

// Contains Edge and Node pair and set of proxy functions to simplify access
// to them.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const {
    return edge_ == other.edge_;
  }
  bool operator!=(const EdgeAndNode& other) const {
    return edge_ != other.edge_;
  }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  // Proxy functions for easier access to node/edge.
  float GetQ(float default_q, float draw_score) const {
    return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q;
  }
  float GetWL(float default_wl) const {
    return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
  }
  float GetD(float default_d) const {
    return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d;
  }
  float GetM(float default_m) const {
    return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
  }
  // N-related getters, from Node (if exists).
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }

  // Whether the node is known to be terminal.
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Bounds GetBounds() const {
    return node_ ? node_->GetBounds()
                 : Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
  }

  // Edge related getters.
  float GetP() const {
    return node_ != nullptr ? node_->GetP() : edge_->GetP();
  }
  Move GetMove(bool flip = false) const {
    return edge_ ? edge_->GetMove(flip) : Move();
  }

  // Returns U = numerator * p / N.
  // Passed numerator is expected to be equal to (cpuct * sqrt(N[parent])).
  float GetU(float numerator) const {
    return numerator * GetP() / (1 + GetNStarted());
  }

  std::string DebugString() const;

 protected:
  // nullptr means that the whole pair is "null". (E.g. when search for a node
  // didn't find anything, or as end iterator signal).
  Edge* edge_ = nullptr;
  // nullptr means that the edge doesn't yet have node extended.
  Node* node_ = nullptr;
};

// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// * Element, pointed by iterator (EdgeAndNode class mainly, but Edge_Iterator
//   is useful too when client wants to call GetOrSpawnNode).
//   It's safe to slice EdgeAndNode off Edge_Iterator.
// It's more customary to have those as three classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized), but
// it's fine if GetOrSpawnNode is called between calls to functions of the
// iterator (e.g. advancing the iterator).
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  // Creates "end()" iterator.
  Edge_Iterator() {}

  // Creates "begin()" iterator.
  Edge_Iterator(LowNode* parent_node)
      : EdgeAndNode(parent_node != nullptr ? parent_node->GetEdges() : nullptr,
                    nullptr),
        parent_node_(parent_node) {
    if (edge_ != nullptr) {
      node_ = parent_node_->GetChildAt(current_idx_);
      total_count_ = parent_node->GetNumEdges();
    }
  }

  // Function to support range interface.
  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
    assert(parent_node_ != nullptr);

    // If it was the last edge in array, become end(), otherwise advance.
    if (++current_idx_ == total_count_) {
      edge_ = nullptr;
    } else {
      ++edge_;
      node_ = parent_node_->GetChildAt(current_idx_);
    }
  }
  Edge_Iterator& operator*() { return *this; }

  // If there is node, return it. Otherwise spawn a new one and return it.
  Node* GetOrSpawnNode() {
    assert(parent_node_ != nullptr);
    if (node_ == nullptr) {
      node_ = parent_node_->InsertChildAt(current_idx_);
    }

    return node_;
  }

 private:
  LowNode* parent_node_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

inline Node::ConstIterator Node::Edges() const { return {this->GetLowNode()}; }
inline Node::Iterator Node::Edges() { return {this->GetLowNode()}; }

// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// It's more customary to have those as two classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized).
template <bool is_const>
class VisitedNode_Iterator {
 public:
  // Creates "end()" iterator.
  VisitedNode_Iterator() {}

  // Creates "begin()" iterator.
  VisitedNode_Iterator(LowNode* parent_node) : parent_node_(parent_node) {
    if (parent_node != nullptr) {
      node_ptr_ = parent_node->GetChildAt(current_idx_);
      total_count_ = parent_node->GetNumEdges();
      if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
        operator++();
      }
    }
  }

  // These are technically wrong, but are usable to compare with end().
  bool operator==(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ == other.node_ptr_;
  }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ != other.node_ptr_;
  }

  // Function to support range interface.
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }

  // Functions to support iterator interface.
  void operator++() {
    assert(parent_node_ != nullptr);

    do {
      ++current_idx_;
      node_ptr_ = parent_node_->GetChildAt(current_idx_);
      // If n started is 0, can jump direct to end due to sorted policy
      // ensuring that each time a new edge becomes best for the first time,
      // it is always the first of the section at the end that has NStarted of
      // 0.
      if (node_ptr_ != nullptr && node_ptr_->GetN() == 0 &&
          node_ptr_->GetNInFlight() == 0) {
        node_ptr_ = nullptr;
        break;
      }
    } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
  }
  Node* operator*() { return node_ptr_; }

 private:
  LowNode* parent_node_ = nullptr;
  // Pointer to current node.
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return {this->GetLowNode()};
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return {this->GetLowNode()};
}

class NodeTree {
 public:
  // Transposition Table (TT) type for holding all normal low nodes in the DAG.
  typedef absl::flat_hash_map<uint64_t, std::unique_ptr<LowNode>>
      TranspositionTable;

  ~NodeTree() { DeallocateTree(); }
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in the tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
  // Returns whether the new position is the same game as the old position (with
  // some moves added). Returns false, if the position is completely different,
  // or if it's shorter than before.
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }
  const std::vector<Move>& GetMoves() const { return moves_; }

  // Look up a low node in the Transposition Table by @hash and return it, or
  // nullptr on failure.
  LowNode* TTFind(uint64_t hash);
  // Get a low node for the @hash from the Transposition Table or create a
  // new low node and insert it into the Transposition Table if it is not there
  // already. Return the low node for the hash.
  std::pair<LowNode*, bool> TTGetOrCreate(uint64_t hash);
  // Evict unused low nodes from the Transposition Table.
  void TTMaintenance();
  // Clear the Transposition Table.
  void TTClear();

  // Add a clone of low @node to special nodes outside of the Transposition
  // Table and return it.
  LowNode* NonTTAddClone(const LowNode& node);

 private:
  void DeallocateTree();

  // Evict unused non-TT low nodes.
  void NonTTMaintenance();
  // Clear non-TT low nodes.
  void NonTTClear();

  // A node which to start search from.
  Node* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
  std::vector<Move> moves_;

  // Transposition Table (TT) for holding references to all normal low nodes in
  // the DAG.
  TranspositionTable tt_;
  // Collection of low nodes that are not fit for Transposition Table due to
  // noise or incomplete information.
  std::vector<std::unique_ptr<LowNode>> non_tt_;
};

}  // namespace lczero
