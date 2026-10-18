// SPDX-License-Identifier: GPL-3.0-only
#include "inez/chess.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using inez::Color;
using inez::Move;
using inez::PieceType;
using inez::Position;
using inez::Termination;

constexpr std::string_view start_fen =
	"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr std::string_view kiwipete =
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

void require(bool condition, std::string_view expression, int line)
{
	if (!condition)
		throw std::runtime_error("line " + std::to_string(line) + ": " +
			std::string(expression));
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

template<class Exception, class Function>
void throws(Function &&function)
{
	bool caught = false;
	try {
		function();
	} catch (const Exception &) {
		caught = true;
	}
	REQUIRE(caught);
}

bool has_move(const Position &position, std::string_view uci)
{
	const auto moves = position.legal_moves();
	return std::any_of(moves.begin(), moves.end(), [uci](const Move &move) {
		return move.uci() == uci;
	});
}

void play(Position &position, std::initializer_list<std::string_view> moves)
{
	for (const auto move : moves)
		position.play_uci(move);
}

void knight_cycle(Position &position)
{
	play(position, {"g1f3", "g8f6", "f3g1", "f6g8"});
}

void fen_roundtrips()
{
	REQUIRE(Position().fen() == start_fen);
	REQUIRE(Position::start() == Position());
	const std::array fixtures = {
		start_fen, kiwipete,
		std::string_view("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 23"),
		std::string_view("7k/8/8/8/8/8/8/KR6 b - - 123 456"),
		std::string_view("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
	};
	for (const auto fen : fixtures) {
		const auto position = Position::from_fen(fen);
		REQUIRE(position.fen() == fen);
		REQUIRE(position.repetition_count() == 1);
		REQUIRE(Position::from_fen(position.fen()) == position);
	}
	const auto clock = Position::from_fen(fixtures[3]);
	REQUIRE(clock.side_to_move() == Color::Black);
	REQUIRE(clock.halfmove_clock() == 123);
	REQUIRE(clock.fullmove_number() == 456);
}

void invalid_fen_and_notation()
{
	const std::vector<std::string_view> invalid = {
		"", "invalid", "8/8/8/8/8/8/8/8 w - - 0 1",
		"7k/8/8/8/8/8/8/K7 w - - 0",
		"7k/8/8/8/8/8/8/K7 w - - 0 1 extra",
		"7k/8/8/8/8/8/8/K8 w - - 0 1",
		"7k/8/8/8/8/8/8/K6x w - - 0 1",
		"7k/8/8/8/8/8/K7 w - - 0 1",
		"7k/8/8/8/8/8/8/K7 x - - 0 1",
		"7k/8/8/8/8/8/8/K7 w X - 0 1",
		"4k3/8/8/8/8/8/8/4K2R w KK - 0 1",
		"4k3/8/8/8/8/8/8/4K3 w K - 0 1", // missing home rook
		"7k/8/8/8/8/8/8/K7 w - - -1 1",
		"7k/8/8/8/8/8/8/K7 w - - 0 0",
		"7k/8/8/8/8/8/8/K7 w - - 18446744073709551616 1",
		"7k/8/8/8/8/8/8/K7 w - e9 0 1",
		"7k/8/8/8/8/8/8/K7 w - e3 0 1",
		"4k3/8/8/8/8/8/8/4K3 w - d6 0 1", // no double-stepped pawn
		"4k3/8/8/3p4/8/8/8/4K3 w - d6 1 1", // EP requires zero clock
		"4k3/3p4/8/3p4/8/8/8/4K3 w - d6 0 1", // source occupied
		"7k/8/8/8/8/8/8/K6P w - - 0 1", // pawn on first rank
		"P6k/8/8/8/8/8/8/K7 w - - 0 1", // pawn on eighth rank
		"7k/8/8/8/8/8/PPPPPPPP/KP6 w - - 0 1", // nine pawns
		"7k/8/8/8/8/8/8/KK6 w - - 0 1",
		"8/8/8/8/8/8/k7/K7 w - - 0 1", // adjacent kings
		"4k3/8/8/8/8/8/4R3/K7 w - - 0 1", // nonmoving king checked
	};
	for (const auto fen : invalid)
		throws<std::invalid_argument>([fen] { (void)Position::from_fen(fen); });
	for (const auto text : {"", "e2e", "e2e4qq", "i2e4", "e0e4", "e7e8k", "e7e8p"})
		throws<std::invalid_argument>([text] { (void)Move::from_uci(text); });
	REQUIRE(Move::from_uci("a7a8n").uci() == "a7a8n");
	REQUIRE(Move::from_uci("e2e4").uci() == "e2e4");
}

void reference_perft()
{
	struct Fixture { std::string_view fen; std::vector<std::uint64_t> nodes; };
	const std::vector<Fixture> fixtures = {
		{start_fen, {1, 20, 400, 8902, 197281}},
		{kiwipete, {1, 48, 2039, 97862}},
		{"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {1, 14, 191, 2812}},
		{"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", {1, 6, 264, 9467}},
	};
	for (const auto &fixture : fixtures) {
		const auto position = Position::from_fen(fixture.fen);
		const auto saved = position;
		for (std::size_t depth = 0; depth < fixture.nodes.size(); ++depth) {
			const auto nodes = position.perft(static_cast<int>(depth));
			if (nodes != fixture.nodes[depth])
				throw std::runtime_error("perft depth " + std::to_string(depth) +
					" for " + std::string(fixture.fen) + ": " + std::to_string(nodes));
			REQUIRE(position == saved);
		}
	}
	const auto start = Position::start();
	throws<std::invalid_argument>([&] { (void)start.perft(-1); });
}

void pins_and_double_check()
{
	const auto pin = Position::from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1");
	REQUIRE(!pin.in_check());
	REQUIRE(!has_move(pin, "e2d2"));
	REQUIRE(has_move(pin, "e2e3"));
	REQUIRE(has_move(pin, "e2e8"));
	const auto double_check = Position::from_fen("4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1");
	REQUIRE(double_check.in_check());
	REQUIRE(!double_check.legal_moves().empty());
	for (const auto move : double_check.legal_moves())
		REQUIRE(double_check.piece_at(move.from).type == PieceType::King);
}

void castling_rules()
{
	for (const auto fen : {
		"4kr2/8/8/8/8/8/8/4K2R w K - 0 1", // f1 attacked
		"4r1k1/8/8/8/8/8/8/4K2R w K - 0 1", // king in check
		"4k1r1/8/8/8/8/8/8/4K2R w K - 0 1", // g1 attacked
		"4k3/8/8/8/8/8/8/4KB1R w K - 0 1", // path occupied
	})
		REQUIRE(!has_move(Position::from_fen(fen), "e1g1"));
	REQUIRE(has_move(Position::from_fen("1r2k3/8/8/8/8/8/8/R3K3 w Q - 0 1"), "e1c1"));
	REQUIRE(has_move(Position::from_fen("r3k3/8/8/8/8/8/8/R3K3 w Q - 0 1"), "e1c1"));
	REQUIRE(!has_move(Position::from_fen("4k3/8/8/8/8/8/8/RN2K3 w Q - 0 1"), "e1c1"));
	REQUIRE(!has_move(Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"), "e1g1"));
	auto capture = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
	capture.play_uci("a1a8");
	REQUIRE(capture.fen() == "R3k2r/8/8/8/8/8/8/4K2R b Kk - 0 1");
	auto moved = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
	play(moved, {"a1a2", "e8e7", "a2a1", "e7e8"});
	REQUIRE(!has_move(moved, "e1c1"));
	auto castle = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
	castle.play_uci("e1g1");
	REQUIRE(castle.fen() == "4k3/8/8/8/8/8/8/5RK1 b - - 1 1");
}

void en_passant_and_canonical_keys()
{
	auto illegal = Position::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1");
	const auto without_illegal = Position::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - - 0 1");
	REQUIRE(!has_move(illegal, "g5f6"));
	REQUIRE(illegal.repetition_key() == without_illegal.repetition_key());
	auto legal = Position::from_fen("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1");
	const auto without_legal = Position::from_fen("4k3/8/8/5pP1/8/8/8/4K3 w - - 0 1");
	REQUIRE(has_move(legal, "g5f6"));
	REQUIRE(!(legal.repetition_key() == without_legal.repetition_key()));
	legal.play_uci("g5f6");
	REQUIRE(legal.fen() == "4k3/8/5P2/8/8/8/8/4K3 b - - 0 1");
	auto raw = Position::start();
	raw.play_uci("e2e4");
	const auto no_raw = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
	REQUIRE(raw.fen().find(" e3 ") != std::string::npos);
	REQUIRE(raw.repetition_key() == no_raw.repetition_key());
	REQUIRE(!(raw == no_raw));
}

void every_promotion()
{
	const auto original = Position::from_fen("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
	const std::array promotions = {
		std::pair{'q', PieceType::Queen}, std::pair{'r', PieceType::Rook},
		std::pair{'b', PieceType::Bishop}, std::pair{'n', PieceType::Knight},
	};
	REQUIRE(!has_move(original, "g7g8"));
	REQUIRE(!has_move(original, "g7h8"));
	for (const auto destination : {std::string("g8"), std::string("h8")}) {
		for (const auto [suffix, type] : promotions) {
			auto position = original;
			const auto text = "g7" + destination + suffix;
			REQUIRE(has_move(position, text));
			position.play_uci(text);
			const auto move = Move::from_uci(text);
			REQUIRE(position.piece_at(move.to).type == type);
			REQUIRE(position.piece_at(move.to).color == Color::White);
			REQUIRE(position.piece_at(move.from).type == PieceType::None);
			REQUIRE(position.halfmove_clock() == 0);
			position.unmake_move();
			REQUIRE(position == original);
		}
	}
}

void mate_stalemate_and_material()
{
	const auto mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
	REQUIRE(mate.in_check());
	REQUIRE(mate.legal_moves().empty());
	REQUIRE(mate.outcome().termination == Termination::Checkmate);
	REQUIRE(mate.outcome().winner == Color::White);
	REQUIRE(!mate.outcome().is_draw());
	const auto stale = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
	REQUIRE(!stale.in_check());
	REQUIRE(stale.legal_moves().empty());
	REQUIRE(stale.outcome().termination == Termination::Stalemate);
	REQUIRE(stale.outcome().is_draw());
	for (const auto fen : {
		"7k/8/8/8/8/8/8/K7 w - - 0 1",
		"7k/8/8/8/8/8/8/KB6 w - - 0 1",
		"7k/8/8/8/8/8/8/KN6 w - - 0 1",
		"5b1k/8/8/8/8/8/8/K1B5 w - - 0 1", // same-color bishops
	}) {
		const auto position = Position::from_fen(fen);
		REQUIRE(position.is_insufficient_material());
		REQUIRE(position.outcome().termination == Termination::InsufficientMaterial);
		REQUIRE(position.outcome().is_draw());
	}
	for (const auto fen : {
		"7k/8/8/8/8/8/8/KNN5 w - - 0 1", // mate possible, though not forced
		"7k/8/8/8/8/8/8/KBN5 w - - 0 1",
		"7k/5b2/8/8/8/8/8/K1B5 w - - 0 1", // opposite-color bishops
		"6nk/8/8/8/8/8/8/KN6 w - - 0 1",
		"7k/8/8/8/8/8/P7/K7 w - - 0 1",
		"7k/8/8/8/8/8/8/KR6 w - - 0 1",
	}) {
		const auto position = Position::from_fen(fen);
		REQUIRE(!position.is_insufficient_material());
		REQUIRE(!position.outcome().is_terminal());
	}
}

void repetition_claims_and_search_identity()
{
	auto position = Position::start();
	REQUIRE(position.repetition_count() == 1);
	REQUIRE(!position.can_claim_threefold_repetition());
	knight_cycle(position);
	REQUIRE(position.repetition_count() == 2);
	REQUIRE(!position.can_claim_threefold_repetition());
	play(position, {"g1f3", "g8f6", "f3g1"});
	REQUIRE(position.repetition_count() == 2);
	REQUIRE(position.can_claim_threefold_repetition()); // by intended f6g8
	REQUIRE(position.outcome().termination == Termination::Ongoing);
	REQUIRE(position.outcome(true).termination == Termination::ThreefoldRepetition);
	position.play_uci("f6g8");
	REQUIRE(position.repetition_count() == 3);
	REQUIRE(position.can_claim_threefold_repetition());
	REQUIRE(position.outcome().termination == Termination::Ongoing);
	REQUIRE(position.outcome(true).is_draw());
	const auto imported = Position::from_fen(position.fen());
	REQUIRE(imported.repetition_count() == 1);
	REQUIRE(imported.repetition_key() == position.repetition_key());
	REQUIRE(imported.halfmove_clock() == position.halfmove_clock());
	REQUIRE(!(imported.search_key() == position.search_key()));
	knight_cycle(position);
	REQUIRE(position.repetition_count() == 4);
	REQUIRE(position.outcome().termination == Termination::Ongoing);
	knight_cycle(position);
	REQUIRE(position.repetition_count() == 5);
	REQUIRE(position.outcome().termination == Termination::FivefoldRepetition);
	REQUIRE(position.outcome(true).termination == Termination::FivefoldRepetition);
	const auto other_clock = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 1 1");
	REQUIRE(other_clock.repetition_key() == Position::start().repetition_key());
	REQUIRE(!(other_clock.search_key() == Position::start().search_key()));
}

void move_count_claims_and_mate_precedence()
{
	const auto at98 = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 98 1");
	REQUIRE(!at98.can_claim_fifty_moves());
	auto at99 = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 99 1");
	REQUIRE(at99.can_claim_fifty_moves()); // intended quiet move reaches 100
	REQUIRE(at99.outcome().termination == Termination::Ongoing);
	REQUIRE(at99.outcome(true).termination == Termination::FiftyMoveRule);
	at99.play_uci("b1b2");
	REQUIRE(at99.halfmove_clock() == 100);
	REQUIRE(at99.can_claim_fifty_moves());
	REQUIRE(at99.outcome().termination == Termination::Ongoing);
	const auto only_pawns = Position::from_fen("1r6/8/8/8/8/8/P1k5/K7 w - - 99 1");
	REQUIRE(!only_pawns.can_claim_fifty_moves());
	for (const auto move : only_pawns.legal_moves())
		REQUIRE(only_pawns.piece_at(move.from).type == PieceType::Pawn);
	auto pawn = Position::from_fen("7k/8/8/8/8/8/P7/KR6 w - - 99 1");
	pawn.play_uci("a2a3");
	REQUIRE(pawn.halfmove_clock() == 0);
	REQUIRE(!pawn.can_claim_fifty_moves());
	auto at149 = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 149 1");
	REQUIRE(at149.outcome().termination == Termination::Ongoing);
	at149.play_uci("b1b2");
	REQUIRE(at149.outcome().termination == Termination::SeventyFiveMoveRule);
	REQUIRE(at149.outcome(true).termination == Termination::SeventyFiveMoveRule);
	auto mate = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 149 1");
	mate.play_uci("g6g7");
	REQUIRE(mate.halfmove_clock() == 150);
	REQUIRE(mate.outcome().termination == Termination::Checkmate);
	REQUIRE(mate.outcome(true).termination == Termination::Checkmate);
	REQUIRE(mate.outcome().winner == Color::White);
}

void exact_undo_and_failed_move_guarantees()
{
	const std::array fixtures = {
		start_fen, kiwipete,
		std::string_view("4k3/8/8/5pP1/8/8/8/4K3 w - f6 0 1"),
		std::string_view("4k2r/6P1/8/8/8/8/8/4K3 w - - 37 10"),
		std::string_view("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 12 8"),
	};
	for (const auto fen : fixtures) {
		auto position = Position::from_fen(fen);
		const auto original = position;
		const auto repetition = position.repetition_key();
		const auto search = position.search_key();
		for (const auto move : position.legal_moves()) {
			position.make_move(move);
			position.unmake_move();
			REQUIRE(position == original);
			REQUIRE(position.repetition_key() == repetition);
			REQUIRE(position.search_key() == search);
		}
	}
	auto position = Position::start();
	std::vector<Position> snapshots{position};
	for (const auto move : {"e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"}) {
		position.play_uci(move);
		snapshots.push_back(position);
	}
	for (std::size_t i = snapshots.size() - 1; i > 0; --i) {
		position.unmake_move();
		REQUIRE(position == snapshots[i - 1]);
	}
	const auto original = position;
	for (const auto move : {"e2e5", "e7e5", "a1a8", "e2e4q", "bad"}) {
		throws<std::invalid_argument>([&] { position.play_uci(move); });
		REQUIRE(position == original);
	}
	throws<std::logic_error>([&] { position.unmake_move(); });
	REQUIRE(position == original);
}

} // namespace

int main()
{
	const std::vector<std::pair<std::string_view, std::function<void()>>> tests = {
		{"FEN round trips", fen_roundtrips},
		{"invalid FEN and notation", invalid_fen_and_notation},
		{"reference perft", reference_perft},
		{"pins and double check", pins_and_double_check},
		{"castling", castling_rules},
		{"en passant and canonical repetition keys", en_passant_and_canonical_keys},
		{"all promotion types", every_promotion},
		{"mate, stalemate, and sound material draws", mate_stalemate_and_material},
		{"repetition claims and search identity", repetition_claims_and_search_identity},
		{"move counts and mate precedence", move_count_claims_and_mate_precedence},
		{"exact undo and failed moves", exact_undo_and_failed_move_guarantees},
	};
	int failed = 0;
	for (const auto &[name, test] : tests) {
		try {
			test();
			std::cout << "PASS " << name << '\n';
		} catch (const std::exception &error) {
			++failed;
			std::cerr << "FAIL " << name << ": " << error.what() << '\n';
		}
	}
	if (failed != 0)
		std::cerr << failed << " test group(s) failed\n";
	return failed == 0 ? 0 : 1;
}
