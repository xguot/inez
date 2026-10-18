// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace inez {

enum class Color { White, Black };
enum class PieceType { None, Pawn, Knight, Bishop, Rook, Queen, King };

struct Piece {
	PieceType type{PieceType::None};
	Color color{Color::White};
	bool operator==(const Piece &) const = default;
};

// Squares are indexed a1 = 0, b1 = 1, ..., h8 = 63.
struct Move {
	int from{-1};
	int to{-1};
	PieceType promotion{PieceType::None};

	[[nodiscard]] static Move from_uci(std::string_view text);
	[[nodiscard]] std::string uci() const;
	bool operator==(const Move &) const = default;
};

// Exact structural identity for FIDE repetition. The en-passant square is
// included only when at least one legal en-passant capture exists.
struct RepetitionKey {
	std::array<Piece, 64> board{};
	Color side_to_move{Color::White};
	std::uint8_t castling_rights{0};
	int en_passant_square{-1};
	bool operator==(const RepetitionKey &) const = default;
};

// Exact, deliberately conservative cache identity. History includes every
// position known since construction/FEN import, including the current one.
// A FEN cannot reconstruct positions played before its import.
struct SearchKey {
	RepetitionKey position{};
	std::uint64_t halfmove_clock{0};
	std::vector<RepetitionKey> history{};
	bool operator==(const SearchKey &) const = default;
};

enum class Termination {
	Ongoing,
	Checkmate,
	Stalemate,
	InsufficientMaterial,
	FiftyMoveRule,
	ThreefoldRepetition,
	SeventyFiveMoveRule,
	FivefoldRepetition,
};

struct Outcome {
	Termination termination{Termination::Ongoing};
	std::optional<Color> winner{};

	[[nodiscard]] bool is_terminal() const noexcept
	{
		return termination != Termination::Ongoing;
	}
	[[nodiscard]] bool is_draw() const noexcept
	{
		return is_terminal() && !winner.has_value();
	}
	bool operator==(const Outcome &) const = default;
};

class Position {
public:
	Position();
	[[nodiscard]] static Position start();
	// Six-field standard-chess FEN; rejects malformed and basic invalid
	// states. Does not prove historical reachability. No Chess960 support.
	[[nodiscard]] static Position from_fen(std::string_view text);
	[[nodiscard]] std::string fen() const;
	[[nodiscard]] std::vector<Move> legal_moves() const;
	// Invalid notation/moves throw std::invalid_argument without mutation.
	void make_move(Move move);
	void play_uci(std::string_view text);
	// Throws std::logic_error when no locally played move can be undone.
	void unmake_move();

	[[nodiscard]] bool in_check() const;
	[[nodiscard]] bool in_check(Color color) const;
	[[nodiscard]] Color side_to_move() const noexcept { return side_; }
	[[nodiscard]] std::uint64_t halfmove_clock() const noexcept { return halfmove_; }
	[[nodiscard]] std::uint64_t fullmove_number() const noexcept { return fullmove_; }
	[[nodiscard]] Piece piece_at(int square) const;
	[[nodiscard]] RepetitionKey repetition_key() const;
	// Deterministic diagnostic hash; collisions never decide repetition/cache equality.
	[[nodiscard]] std::uint64_t repetition_hash() const;
	[[nodiscard]] int repetition_count() const;
	[[nodiscard]] SearchKey search_key() const;
	// Sound standard material cases only, not a general dead-position solver.
	[[nodiscard]] bool is_insufficient_material() const;
	// Claims include an eligible intended legal move as well as the current
	// position. Automatic terminations are reported by outcome().
	[[nodiscard]] bool can_claim_fifty_moves() const;
	[[nodiscard]] bool can_claim_threefold_repetition() const;
	[[nodiscard]] Outcome outcome(bool claim_draw = false) const;
	// Counts legal move paths; ignores draw adjudication, as standard perft does.
	[[nodiscard]] std::uint64_t perft(int depth) const;

	// Includes clocks, raw FEN en-passant state, known history, and undo stack.
	bool operator==(const Position &) const = default;

private:
	struct EmptyTag {};
	explicit Position(EmptyTag) {}
	struct Undo {
		std::array<Piece, 64> board{};
		Color side{Color::White};
		std::uint8_t castling{0};
		int ep{-1};
		std::uint64_t halfmove{0};
		std::uint64_t fullmove{1};
		bool operator==(const Undo &) const = default;
	};

	std::array<Piece, 64> board_{};
	Color side_{Color::White};
	std::uint8_t castling_{0};
	int ep_{-1};
	std::uint64_t halfmove_{0};
	std::uint64_t fullmove_{1};
	std::vector<RepetitionKey> history_{};
	std::vector<Undo> undo_{};

	[[nodiscard]] std::vector<Move> pseudo_legal_moves() const;
	[[nodiscard]] bool has_legal_en_passant() const;
	void apply_unchecked(Move move);
	[[nodiscard]] std::uint64_t perft_impl(int depth);
};

} // namespace inez
