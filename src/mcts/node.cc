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

#include "mcts/node.h"

#include <absl/algorithm/container.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "utils/exception.h"
#include "utils/hashcat.h"
#include "utils/numa.h"

namespace lczero {

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Mirror();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).

// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " p_: " << p_ << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// LowNode + Node
/////////////////////////////////////////////////////////////////////////

Node& Node::operator=(Node&& move_from) {  // Race expected.
  assert(move_from.Realized());

  // Try to lock others out.
  uint16_t expected_index = kMagicIndexConstructed;
  if (!index_.compare_exchange_strong(expected_index, kMagicIndexAssigned)) {
    // Someone was faster, wait for them to finish.
    do {
      // EMPTY
    } while (!Realized());

    return *this;
  }

  wl_ = move_from.wl_;

  // Move low node over without updating it using Get/SetLowNode.
  low_node_ = move_from.low_node_;
  move_from.low_node_ = nullptr;

  d_ = move_from.d_;
  m_ = move_from.m_;
  n_ = move_from.n_;
  n_in_flight_.store(move_from.n_in_flight_.load(std::memory_order_acquire),
                     std::memory_order_release);

  edge_ = move_from.edge_;

  // index_ is updated last.

  terminal_type_ = move_from.terminal_type_;
  lower_bound_ = move_from.lower_bound_;
  upper_bound_ = move_from.upper_bound_;

  // Unlock node and make it Realized().
  index_.store(move_from.index_.load(std::memory_order_acquire),
               std::memory_order_release);

  return *this;
}

void Node::Reset() {  // No race expected.
  wl_ = 0.0f;

  UnsetLowNode();

  d_ = 0.0f;
  m_ = 0.0f;
  n_ = 0;
  n_in_flight_ = 0;

  edge_ = Edge();

  index_.store(kMagicIndexConstructed, std::memory_order_relaxed);

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  repetition_ = false;
}

void Node::Trim() {  // No race expected.
  wl_ = 0.0f;

  UnsetLowNode();

  d_ = 0.0f;
  m_ = 0.0f;
  n_ = 0;
  n_in_flight_ = 0;

  // edge_

  // index_

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  repetition_ = false;
}

Node* Node::GetChild() const {
  if (!low_node_) return nullptr;
  return low_node_->GetChild();
}

bool Node::HasChildren() const { return low_node_ && low_node_->HasChildren(); }

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += node->GetP();
  return sum;
}

uint32_t Node::GetNInFlight() const {
  return n_in_flight_.load(std::memory_order_acquire);
}

uint32_t Node::GetChildrenVisits() const {
  return low_node_ ? low_node_->GetChildrenVisits() : 0;
}

uint32_t Node::GetTotalVisits() const {
  return low_node_ ? low_node_->GetN() : 0;
}

const Edge& LowNode::GetEdgeAt(uint16_t index) const { return edges_[index]; }

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " <Node> This:" << this << " LowNode:" << low_node_
      << " Index:" << index_ << " Move:" << GetMove().as_string()
      << " P:" << GetP() << " WL:" << wl_ << " D:" << d_ << " M:" << m_
      << " N:" << n_ << " N_:" << n_in_flight_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2;
  return oss.str();
}

std::string LowNode::DebugString() const {
  std::ostringstream oss;
  oss << " <LowNode> This:" << this << " Edges:" << edges_.get()
      << " NumEdges:" << static_cast<int>(num_edges_)
      << " AllocatedChildren:" << allocated_children_ << " WL:" << wl_
      << " D:" << d_ << " M:" << m_ << " N:" << n_ << " NP:" << num_parents_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << " IsTransposition:" << is_transposition;
  return oss.str();
}

void Edge::SortEdges(Edge* edges, int num_edges) {
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges, (edges + num_edges),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void LowNode::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::MakeNotTerminal(const Node* node) {
  assert(edges_);
  if (!IsTerminal()) return;

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  n_ = 0;
  wl_ = 0.0;
  d_ = 0.0;
  m_ = 0.0;

  // Include children too.
  if (node->GetNumEdges() > 0) {
    for (const auto& child : node->Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
        m_ += child.GetM(0.0f) * n;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= n_;
    d_ /= n_;
    m_ /= n_;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

uint8_t Node::GetNumEdges() const {
  return low_node_ ? low_node_->GetNumEdges() : 0;
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    SetP(0.0f);
  }

  assert(WLDMInvariantsHold());
}

void Node::MakeNotTerminal(bool also_low_node) {
  // At least one of node and low node pair needs to be a terminal.
  if (!IsTerminal() &&
      (!also_low_node || !low_node_ || !low_node_->IsTerminal()))
    return;

  terminal_type_ = Terminal::NonTerminal;
  repetition_ = false;
  if (low_node_) {  // Two-fold or derived terminal.
    // Revert low node first.
    if (also_low_node && low_node_) low_node_->MakeNotTerminal(this);

    auto [lower_bound, upper_bound] = low_node_->GetBounds();
    lower_bound_ = -upper_bound;
    upper_bound_ = -lower_bound;
    n_ = low_node_->GetN();
    wl_ = -low_node_->GetWL();
    d_ = low_node_->GetD();
    m_ = low_node_->GetM() + 1;
  } else {  // Real terminal.
    lower_bound_ = GameResult::BLACK_WON;
    upper_bound_ = GameResult::WHITE_WON;
    n_ = 0.0f;
    wl_ = 0.0f;
    d_ = 0.0f;
    m_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ > 0) {
    n_in_flight_.fetch_add(1, std::memory_order_acq_rel);
  } else {
    uint32_t expected_n_if_flight_ = 0;
    if (!n_in_flight_.compare_exchange_strong(expected_n_if_flight_, 1,
                                              std::memory_order_acq_rel)) {
      return false;
    }
  }

  return true;
}

void Node::CancelScoreUpdate(uint32_t multivisit) {
  assert(GetNInFlight() >= (uint32_t)multivisit);
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

void LowNode::FinalizeScoreUpdate(float v, float d, float m,
                                  uint32_t multivisit) {
  assert(edges_);
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);

  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
}

void LowNode::AdjustForTerminal(float v, float d, float m,
                                uint32_t multivisit) {
  assert(static_cast<uint32_t>(multivisit) <= n_);

  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;

  assert(WLDMInvariantsHold());
}

void Node::FinalizeScoreUpdate(float v, float d, float m, uint32_t multivisit) {
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);

  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
  // Decrement virtual loss.
  assert(GetNInFlight() >= (uint32_t)multivisit);
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

void Node::AdjustForTerminal(float v, float d, float m, uint32_t multivisit) {
  assert(static_cast<uint32_t>(multivisit) <= n_);

  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;

  assert(WLDMInvariantsHold());
}

void Node::IncrementNInFlight(uint32_t multivisit) {
  n_in_flight_.fetch_add(multivisit, std::memory_order_acq_rel);
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save) const {
  // Sometime we have no graph yet or a reverted terminal without low node.
  if (low_node_) low_node_->ReleaseChildrenExceptOne(node_to_save);
}

void Node::SetLowNode(LowNode* low_node) {
  assert(!low_node_);
  low_node->AddParent();
  low_node_ = low_node;
}
void Node::UnsetLowNode() {
  if (low_node_) low_node_->RemoveParent();
  low_node_ = nullptr;
}

static std::string PtrToNodeName(const void* ptr) {
  std::ostringstream oss;
  oss << "n_" << ptr;
  return oss.str();
}

std::string LowNode::DotNodeString() const {
  std::ostringstream oss;
  oss << PtrToNodeName(this) << " ["
      << "shape=box";
  // Adjust formatting to limit node size.
  oss << std::fixed << std::setprecision(3);
  oss << ",label=\""     //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\lD=" << d_ << "\\lM=" << m_ << "\\lN=" << n_ << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::showpos << std::setprecision(5);
  oss << ",tooltip=\""   //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nNP=" << num_parents_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << "\\nIsTransposition=" << is_transposition  //
      << std::noshowpos                             //
      << "\\n\\nThis=" << this << "\\nEdges=" << edges_.get()
      << "\\nNumEdges=" << static_cast<int>(num_edges_)
      << "\\nAllocatedChildren=" << allocated_children_ << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotEdgeString(bool as_opponent, const LowNode* parent) const {
  std::ostringstream oss;
  oss << (parent == nullptr ? "top" : PtrToNodeName(parent)) << " -> "
      << (low_node_ ? PtrToNodeName(low_node_) : PtrToNodeName(this)) << " [";
  oss << "label=\""
      << (parent == nullptr ? "N/A" : GetMove(as_opponent).as_string())
      << "\\lN=" << n_ << "\\lN_=" << n_in_flight_;
  oss << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::setprecision(5);
  oss << ",labeltooltip=\""
      << "P=" << (parent == nullptr ? 0.0f : GetP())  //
      << std::showpos                                 //
      << "\\nWL= " << wl_                             //
      << std::noshowpos                               //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nN_=" << n_in_flight_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << "\\n\\nThis=" << this  //
      << std::noshowpos                                               //
      << "\\nLowNode=" << low_node_ << "\\nParent=" << parent
      << "\\nIndex=" << index_ << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotGraphString(bool as_opponent) const {
  std::ostringstream oss;
  std::unordered_set<const LowNode*> seen;
  std::list<std::pair<const Node*, bool>> unvisited_fifo;

  oss << "strict digraph {" << std::endl;
  oss << "edge ["
      << "headport=n"
      << ",tooltip=\" \""  // Remove default tooltips from edge parts.
      << "];" << std::endl;
  oss << "node ["
      << "shape=point"    // For fake nodes.
      << ",style=filled"  // Show tooltip everywhere on the node.
      << ",fillcolor=ivory"
      << "];" << std::endl;
  oss << "ranksep=" << 4.0f * std::log10(GetN()) << std::endl;

  oss << DotEdgeString(!as_opponent) << std::endl;
  if (low_node_) {
    seen.insert(low_node_);
    unvisited_fifo.push_back(std::pair(this, as_opponent));
  }

  while (!unvisited_fifo.empty()) {
    auto [parent_node, parent_as_opponent] = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    auto parent_low_node = parent_node->GetLowNode();
    seen.insert(parent_low_node);
    oss << parent_low_node->DotNodeString() << std::endl;

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      oss << child->DotEdgeString(parent_as_opponent) << std::endl;
      auto child_low_node = child->GetLowNode();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(std::pair(child, !parent_as_opponent));
      }
    }
  }

  oss << "}" << std::endl;

  return oss.str();
}

bool Node::ZeroNInFlight() const {
  std::unordered_set<const LowNode*> seen;
  std::list<const Node*> unvisited_fifo;
  size_t nonzero_node_count = 0;

  if (GetNInFlight() > 0) {
    std::cerr << DebugString() << std::endl;
    ++nonzero_node_count;
  }
  if (low_node_) {
    seen.insert(low_node_);
    unvisited_fifo.push_back(this);
  }

  while (!unvisited_fifo.empty()) {
    auto parent_node = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      if (child->GetNInFlight() > 0) {
        std::cerr << child->DebugString() << std::endl;
        ++nonzero_node_count;
      }

      auto child_low_node = child->GetLowNode();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(child);
      }
    }
  }

  if (nonzero_node_count > 0) {
    std::cerr << "GetNInFlight() is nonzero on " << nonzero_node_count
              << " nodes" << std::endl;
    return false;
  }

  return true;
}

void Node::SortEdges() const {
  assert(low_node_);
  low_node_->SortEdges();
}

void LowNode::ReleaseChildren() {  // No race expected.
  auto allocated = allocated_children_.load(std::memory_order_relaxed);
  std::allocator<Node> alloc;

#if STATIC_CHILDREN_ARRAY_SIZE > 0
  // Reset all statically allocated children.
  for (uint16_t i = 0; i < kStaticChildrenArraySize; ++i) {
    static_children_[i].Reset();
  }
#endif

  // Free all arrays for dynamically allocated children.
  for (size_t i = 0; i < kDynamicChildrenArrayCount; ++i) {
    auto array =
        dynamic_children_[i].exchange(nullptr, std::memory_order_relaxed);
    if (array != nullptr) {
      // All but the last array are of a known constant size.
      if (i == kDynamicChildrenArrayCount - 1) {
        alloc.deallocate(array,
                         allocated - kDynamicChildrenArrayKnownTotalSize);
      } else {
        alloc.deallocate(array, kDynamicChildrenArraySizes[i]);
      }
    }
  }

  allocated_children_.store(kStaticChildrenArraySize,
                            std::memory_order_relaxed);
}

void LowNode::ReleaseChildrenExceptOne(
    Node* child_to_save) {  // No race expected.
  assert(child_to_save != nullptr);
  // Save node's content.
  Node saved_child = std::move(*child_to_save);
  // Release all children to maybe save memory.
  ReleaseChildren();
  // Recreate child.
  auto new_child = InsertChildAt(saved_child.Index(), false);
  *new_child = std::move(saved_child);
}

Node* LowNode::GetChild() {
  for (uint16_t i = 0; i < allocated_children_; ++i) {
    auto child = GetChildAt(i);
    if (child) return child;
  }

  return nullptr;
}

Node* LowNode::FindPlaceOf(uint16_t index) {
#if STATIC_CHILDREN_ARRAY_SIZE > 0
  // Find the right child group for the index.
  if (index < kStaticChildrenArraySize) {
    return &static_children_[index];
  }
#endif

  // constexpr
  size_t i = 0;
  while (i < kDynamicChildrenArrayCount - 1 &&
         index >= kDynamicChildrenArrayEnds[i]) {
    ++i;
  }

  auto children = dynamic_children_[i].load(std::memory_order_acquire);
  return &children[index - kDynamicChildrenArrayStarts[i]];
}

Node* LowNode::GetChildAt(uint16_t index) {  // Race expected.
  // Make sure we are looking for a possible realized child.
  if (index >= allocated_children_.load(std::memory_order_acquire))
    return nullptr;

  // Find place with child.
  Node* child = FindPlaceOf(index);

  // Return child, if realized.
  if (child->Realized()) return child;

  return nullptr;
}

void LowNode::Allocate(uint16_t size, uint16_t* already_allocated,
                       std::atomic<Node*>* children) {  // Race expected.
  Node* expected_array = nullptr;
  auto target_allocated = *already_allocated + size;

  // Optimistically allocate and initialize a new array.
  std::allocator<Node> alloc;
  auto array = alloc.allocate(size);
  for (uint16_t i = 0; i < size; ++i) {
    new (&array[i]) Node();
  }

  // Setting array either succeeds or someone else was faster.
  if (children->compare_exchange_strong(expected_array, array,
                                        std::memory_order_acq_rel)) {
    // No one should be updating allocated children count as they would be
    // trying to set the same array and failing.
    allocated_children_.store(target_allocated, std::memory_order_release);
    *already_allocated = target_allocated;
  } else {
    // Free unused new array.
    alloc.deallocate(array, size);

    // Have to wait until allocated children count is updated. Others might even
    // be able to allocate more than one array.
    do {
      *already_allocated = allocated_children_.load(std::memory_order_acquire);
    } while (*already_allocated < target_allocated);
  }
}

Node* LowNode::InsertChildAt(uint16_t index, bool init) {  // Race expected.
  assert(edges_);
  assert(index < num_edges_);

  // Allocate memory for all missing child arrays needed.
  auto allocated = allocated_children_.load(std::memory_order_acquire);
  if (index >= allocated) {
    // Allocate memory for all missing child arrays needed.
    size_t i;
    for (i = 0; i < kDynamicChildrenArrayCount - 1 && index >= allocated; ++i) {
      if (allocated < kDynamicChildrenArrayEnds[i]) {
        Allocate(kDynamicChildrenArraySizes[i], &allocated,
                 &dynamic_children_[i]);
      }
    }

    if (i == kDynamicChildrenArrayCount - 1 && index >= allocated) {
      Allocate(num_edges_ - kDynamicChildrenArrayKnownTotalSize, &allocated,
               &dynamic_children_[i]);
    }
  }

  // Find place with child.
  Node* child = FindPlaceOf(index);

  // Realize child if needed.
  if (init && !child->Realized()) {
    // This either succeeds or someone else is faster and does the same.
    *child = Node(edges_[index], index);
  }

  return child;
}

static constexpr float wld_tolerance = 0.000001f;
static constexpr float m_tolerance = 0.000001f;

static bool WLDMInvariantsHold(float wl, float d, float m) {
  return -(1.0f + wld_tolerance) < wl && wl < (1.0f + wld_tolerance) &&  //
         -(0.0f + wld_tolerance) < d && d < (1.0f + wld_tolerance) &&    //
         -(0.0f + m_tolerance) < m &&                                    //
         std::abs(wl + d) < (1.0f + wld_tolerance);
}

bool Node::WLDMInvariantsHold() const {
  if (lczero::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

bool LowNode::WLDMInvariantsHold() const {
  if (lczero::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

void NodeTree::MakeMove(Move move) {
  if (HeadPosition().IsBlackToMove()) move.Mirror();
  const auto& board = HeadPosition().GetBoard();

  // Find edge for @move, if it exists.
  Node* new_head = nullptr;
  for (auto& n : current_head_->Edges()) {
    if (board.IsSameMove(n.GetMove(), move)) {
      new_head = n.GetOrSpawnNode();
      // Ensure head is not terminal, so search can extend or visit children of
      // "terminal" positions, e.g., WDL hits, converted terminals, 3-fold draw.
      if (new_head->IsTerminal()) new_head->MakeNotTerminal();
      break;
    }
  }
  move = board.GetModernMove(move);
  // Remove edges that will not be needed any more.
  current_head_->ReleaseChildrenExceptOne(new_head);
  new_head = current_head_->GetChild();
  // Use an existing edge for @move or make a new one.
  if (new_head) {
    current_head_ = new_head;
  } else {
    non_tt_.emplace_back(
        std::make_unique<LowNode>(MoveList({move}), static_cast<uint16_t>(0)));
    current_head_->SetLowNode(non_tt_.back().get());
    current_head_ = current_head_->GetChild();
  }
  history_.Append(move);
  moves_.push_back(move);
}

void NodeTree::TrimTreeAtHead() {
  current_head_->Trim();
  // Free unused non-TT low nodes.
  NonTTMaintenance();
}

bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ &&
      (history_.Starting().GetBoard() != starting_board ||
       history_.Starting().GetRule50Ply() != no_capture_ply)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>();
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));
  moves_.clear();

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head) TrimTreeAtHead();
  return seen_old_head;
}

void NodeTree::DeallocateTree() {
  gamebegin_node_.reset();
  current_head_ = nullptr;
  // Free all nodes.
  NonTTClear();
  TTClear();
}

LowNode* NodeTree::TTFind(uint64_t hash) {
  auto tt_iter = tt_.find(hash);
  if (tt_iter != tt_.end()) {
    return tt_iter->second.get();
  } else {
    return nullptr;
  }
}

std::pair<LowNode*, bool> NodeTree::TTGetOrCreate(uint64_t hash) {
  auto [tt_iter, is_tt_miss] = tt_.insert({hash, std::make_unique<LowNode>()});
  return {tt_iter->second.get(), is_tt_miss};
}

void NodeTree::TTMaintenance() {
  absl::erase_if(
      tt_, [](const auto& item) { return item.second->GetNumParents() == 0; });
}

void NodeTree::TTClear() {
  absl::c_for_each(tt_,
                   [](const auto& item) { item.second->ReleaseChildren(); });
  // A single low node might still be attached to the current head.
  TTMaintenance();
}

LowNode* NodeTree::NonTTAddClone(const LowNode& node) {
  non_tt_.push_back(std::make_unique<LowNode>(node));
  return non_tt_.back().get();
}

void NodeTree::NonTTMaintenance() {
  // Find the first parentless low node and remove all low nodes from this low
  // node on.
  auto it = non_tt_.cbegin();
  while (it != non_tt_.cend() && (*it)->GetNumParents() > 0) ++it;
  non_tt_.erase(it, non_tt_.cend());
}

void NodeTree::NonTTClear() {
  absl::c_for_each(non_tt_, [](const auto& item) { item->ReleaseChildren(); });
  // A single low node might still be attached to the game begin node.
  NonTTMaintenance();
}

}  // namespace lczero
