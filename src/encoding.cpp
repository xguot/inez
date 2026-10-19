// SPDX-License-Identifier: GPL-3.0-only
#include "inez/encoding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inez::encoding {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
	"the encoding contract requires IEEE 754 float32");

std::size_t unchecked_move_index(Move move, Color perspective)
{
	const int from = orient_square(move.from, perspective);
	const int to = orient_square(move.to, perspective);
	if (move.promotion == PieceType::None || move.promotion == PieceType::Queen)
		return static_cast<std::size_t>(from * 64 + to);

	std::size_t piece_index = 0;
	switch (move.promotion) {
	case PieceType::Knight: piece_index = 0; break;
	case PieceType::Bishop: piece_index = 1; break;
	case PieceType::Rook: piece_index = 2; break;
	default: throw std::logic_error("legal move has an invalid promotion type");
	}
	if (from / 8 != 6 || to / 8 != 7)
		throw std::logic_error("legal underpromotion has invalid normalized ranks");
	return 4096 + piece_index * 64 + static_cast<std::size_t>((from % 8) * 8 + to % 8);
}

struct LegalEncoding {
	std::vector<Move> moves;
	std::vector<std::size_t> indices;
};

LegalEncoding encode_legal(const Position &position)
{
	LegalEncoding result{position.legal_moves(), {}};
	result.indices.reserve(result.moves.size());
	PolicyMask seen{};
	for (Move move : result.moves) {
		const auto index = unchecked_move_index(move, position.side_to_move());
		if (index >= policy_size || seen[index])
			throw std::logic_error("legal moves do not have unique policy indices");
		seen[index] = 1;
		result.indices.push_back(index);
	}
	return result;
}

template<typename T>
std::vector<double> softmax_impl(const Position &position, std::span<const T> logits)
{
	if (logits.size() != policy_size)
		throw std::invalid_argument("policy logits have incompatible dimensions");
	const auto indices = legal_indices(position);
	if (indices.empty())
		return {};
	double maximum = -std::numeric_limits<double>::infinity();
	for (const auto index : indices) {
		const double logit = static_cast<double>(logits[index]);
		if (!std::isfinite(logit))
			throw std::invalid_argument("legal policy logits must be finite");
		maximum = std::max(maximum, logit);
	}
	std::vector<double> probabilities;
	probabilities.reserve(indices.size());
	double total = 0.0;
	for (const auto index : indices) {
		const double weight = std::exp(static_cast<double>(logits[index]) - maximum);
		probabilities.push_back(weight);
		total += weight;
	}
	for (double &probability : probabilities)
		probability /= total;
	return probabilities;
}

} // namespace

void validate_metadata(const TensorMetadata &metadata)
{
	if (metadata != TensorMetadata{})
		throw std::invalid_argument("incompatible encoding contract version or dimensions");
}

int orient_square(int square, Color current_side)
{
	if (square < 0 || square >= 64 ||
	    (current_side != Color::White && current_side != Color::Black))
		throw std::invalid_argument("invalid encoding square or perspective");
	return current_side == Color::White ? square : square ^ 56;
}

Features encode_features(const Position &position)
{
	Features tensor{};
	const Color perspective = position.side_to_move();
	const SearchKey context = position.search_key();
	const auto &history = context.history;
	for (std::size_t frame = 0; frame < history_frames && frame < history.size(); ++frame) {
		const std::size_t history_index = history.size() - 1 - frame;
		const RepetitionKey &state = history[history_index];
		const auto prefix_end = history.begin() + static_cast<std::ptrdiff_t>(history_index);
		const float repeated = std::find(history.begin(), prefix_end, state) != prefix_end ? 1.0F : 0.0F;
		const std::size_t channel_base = frame * 14;
		for (int square = 0; square < 64; ++square) {
			const auto normalized = static_cast<std::size_t>(orient_square(square, perspective));
			const std::size_t offset = normalized * feature_count + channel_base;
			const Piece piece = state.board[square];
			if (piece.type != PieceType::None) {
				const auto piece_channel = static_cast<std::size_t>(piece.type) - 1;
				const std::size_t relative_color = piece.color == perspective ? 0 : 6;
				tensor[offset + relative_color + piece_channel] = 1.0F;
			}
			tensor[offset + 12] = repeated;
			tensor[offset + 13] = 1.0F;
		}
	}

	const auto rights = context.position.castling_rights;
	const std::uint8_t own_k = perspective == Color::White ? 1 : 4;
	const std::uint8_t own_q = perspective == Color::White ? 2 : 8;
	const std::uint8_t opponent_k = perspective == Color::White ? 4 : 1;
	const std::uint8_t opponent_q = perspective == Color::White ? 8 : 2;
	const float halfmove = static_cast<float>(std::min<std::uint64_t>(context.halfmove_clock, 150)) / 150.0F;
	for (std::size_t square = 0; square < square_count; ++square) {
		const std::size_t offset = square * feature_count;
		tensor[offset + 112] = (rights & own_k) != 0 ? 1.0F : 0.0F;
		tensor[offset + 113] = (rights & own_q) != 0 ? 1.0F : 0.0F;
		tensor[offset + 114] = (rights & opponent_k) != 0 ? 1.0F : 0.0F;
		tensor[offset + 115] = (rights & opponent_q) != 0 ? 1.0F : 0.0F;
		tensor[offset + 117] = halfmove;
		tensor[offset + 118] = perspective == Color::Black ? 1.0F : 0.0F;
	}
	if (context.position.en_passant_square != -1) {
		const auto ep = static_cast<std::size_t>(orient_square(context.position.en_passant_square, perspective));
		tensor[ep * feature_count + 116] = 1.0F;
	}
	return tensor;
}

std::size_t encode_move(const Position &position, Move move)
{
	const auto legal = encode_legal(position);
	const auto found = std::find(legal.moves.begin(), legal.moves.end(), move);
	if (found == legal.moves.end())
		throw std::invalid_argument("policy encoding requires a legal move");
	return legal.indices[static_cast<std::size_t>(found - legal.moves.begin())];
}

Move decode_move(const Position &position, std::size_t index)
{
	if (index >= policy_size)
		throw std::invalid_argument("policy index out of range");
	const auto legal = encode_legal(position);
	const auto found = std::find(legal.indices.begin(), legal.indices.end(), index);
	if (found == legal.indices.end())
		throw std::invalid_argument("policy index does not identify a legal move");
	return legal.moves[static_cast<std::size_t>(found - legal.indices.begin())];
}

std::vector<std::size_t> legal_indices(const Position &position)
{
	return encode_legal(position).indices;
}

PolicyMask legal_mask(const Position &position)
{
	PolicyMask mask{};
	for (const auto index : legal_indices(position))
		mask[index] = 1;
	return mask;
}

std::vector<double> legal_softmax(const Position &position, std::span<const float> logits)
{
	return softmax_impl(position, logits);
}

std::vector<double> legal_softmax(const Position &position, std::span<const double> logits)
{
	return softmax_impl(position, logits);
}

void validate_wdl(const Wdl &wdl)
{
	double total = 0.0;
	for (double probability : wdl) {
		if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
			throw std::invalid_argument("WDL probabilities must be finite and in [0, 1]");
		total += probability;
	}
	if (std::abs(total - 1.0) > 1e-6)
		throw std::invalid_argument("WDL probabilities must sum to one");
}

double wdl_value(const Wdl &wdl)
{
	validate_wdl(wdl);
	return wdl[0] - wdl[2];
}

} // namespace inez::encoding
