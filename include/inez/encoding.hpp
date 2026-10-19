// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "inez/chess.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace inez::encoding {

inline constexpr std::uint32_t contract_version = 1;
inline constexpr std::size_t history_frames = 8;
inline constexpr std::size_t square_count = 64;
inline constexpr std::size_t feature_count = 119;
inline constexpr std::size_t policy_size = 4288;
inline constexpr std::size_t feature_values = square_count * feature_count;

// Square-major float32: tensor[square * feature_count + channel].
using Features = std::array<float, feature_values>;
using PolicyMask = std::array<std::uint8_t, policy_size>;
// Probabilities in [win, draw, loss] order, for the current side to move.
using Wdl = std::array<double, 3>;

struct TensorMetadata {
	std::uint32_t version{contract_version};
	std::size_t squares{square_count};
	std::size_t features{feature_count};
	std::size_t policy{policy_size};
	bool operator==(const TensorMetadata &) const = default;
};

// Rejects incompatible versions or dimensions with std::invalid_argument.
void validate_metadata(const TensorMetadata &metadata);

// Every frame uses the CURRENT side's perspective: White unchanged, Black
// rank-reflected (sq ^ 56), with relative ours/theirs piece channels.
[[nodiscard]] int orient_square(int square, Color current_side);
[[nodiscard]] Features encode_features(const Position &position);

// Requires a current legal move. Normal moves and queen promotions use
// from*64+to. N/B/R underpromotions use the reserved file-pair indices.
[[nodiscard]] std::size_t encode_move(const Position &position, Move move);
// Finds the unique legal move with this index; rejects unused/illegal indices.
[[nodiscard]] Move decode_move(const Position &position, std::size_t index);

// Indices and priors retain Position::legal_moves() order.
[[nodiscard]] std::vector<std::size_t> legal_indices(const Position &position);
[[nodiscard]] PolicyMask legal_mask(const Position &position);
// Full-policy logits must have policy_size entries. Only legal entries must
// be finite. Positions with no legal moves return empty priors. Like the
// rules API, these helpers retain legal moves after an automatic draw;
// callers must adjudicate terminal outcomes before evaluating a position.
[[nodiscard]] std::vector<double> legal_softmax(
	const Position &position, std::span<const float> logits);
[[nodiscard]] std::vector<double> legal_softmax(
	const Position &position, std::span<const double> logits);

// Finite probabilities in [0, 1], summing to one within 1e-6.
void validate_wdl(const Wdl &wdl);
[[nodiscard]] double wdl_value(const Wdl &wdl);

} // namespace inez::encoding
