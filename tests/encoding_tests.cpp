// SPDX-License-Identifier: GPL-3.0-only
#include "inez/encoding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using inez::Color;
using inez::Move;
using inez::Position;
namespace encoding = inez::encoding;

void require(bool condition, std::string_view expression, int line)
{
	if (!condition)
		throw std::runtime_error("line " + std::to_string(line) + ": " + std::string(expression));
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

bool near(double lhs, double rhs, double tolerance = 1e-12)
{
	return std::abs(lhs - rhs) <= tolerance;
}

float feature(const encoding::Features &tensor, int square, std::size_t channel)
{
	return tensor[static_cast<std::size_t>(square) * encoding::feature_count + channel];
}

void knight_cycle(Position &position)
{
	for (const auto text : {"g1f3", "g8f6", "f3g1", "f6g8"})
		position.play_uci(text);
}

void dimensions_and_padding()
{
	static_assert(encoding::contract_version == 1);
	static_assert(encoding::history_frames == 8);
	static_assert(encoding::feature_count == 119);
	static_assert(encoding::policy_size == 4288);
	static_assert(encoding::feature_values == 7616);
	const auto tensor = encoding::encode_features(Position::start());
	REQUIRE(tensor.size() == 7616);
	REQUIRE(feature(tensor, 0, 3) == 1.0F); // ours: rook a1
	REQUIRE(feature(tensor, 4, 5) == 1.0F); // ours: king e1
	REQUIRE(feature(tensor, 56, 9) == 1.0F); // theirs: rook a8
	REQUIRE(feature(tensor, 60, 11) == 1.0F); // theirs: king e8
	for (int square = 0; square < 64; ++square) {
		REQUIRE(feature(tensor, square, 12) == 0.0F);
		REQUIRE(feature(tensor, square, 13) == 1.0F);
		for (std::size_t channel = 14; channel < 112; ++channel)
			REQUIRE(feature(tensor, square, channel) == 0.0F);
		for (std::size_t channel = 112; channel < 116; ++channel)
			REQUIRE(feature(tensor, square, channel) == 1.0F);
		for (std::size_t channel = 116; channel < 119; ++channel)
			REQUIRE(feature(tensor, square, channel) == 0.0F);
	}
	REQUIRE(std::count(tensor.begin(), tensor.end(), 1.0F) == 32 + 64 + 4 * 64);
	encoding::validate_metadata({});
	for (int field = 0; field < 4; ++field) {
		encoding::TensorMetadata metadata;
		if (field == 0) ++metadata.version;
		if (field == 1) ++metadata.squares;
		if (field == 2) ++metadata.features;
		if (field == 3) ++metadata.policy;
		throws<std::invalid_argument>([&] { encoding::validate_metadata(metadata); });
	}
}

void current_perspective_for_every_frame()
{
	auto position = Position::start();
	position.play_uci("e2e4");
	const auto tensor = encoding::encode_features(position);
	REQUIRE(position.side_to_move() == Color::Black);
	REQUIRE(encoding::orient_square(56, Color::Black) == 0);
	REQUIRE(encoding::orient_square(0, Color::Black) == 56);
	REQUIRE(encoding::orient_square(27, Color::White) == 27);
	REQUIRE(feature(tensor, 0, 3) == 1.0F); // current black rook a8 -> a1
	REQUIRE(feature(tensor, 36, 6) == 1.0F); // current white pawn e4 -> e5
	REQUIRE(feature(tensor, 52, 6) == 0.0F); // white pawn has left e2 -> e7
	REQUIRE(feature(tensor, 1, 14 + 1) == 1.0F); // past black knight b8 -> b1, ours
	REQUIRE(feature(tensor, 52, 14 + 6) == 1.0F); // past white e2 -> e7, theirs
	for (int square = 0; square < 64; ++square) {
		REQUIRE(feature(tensor, square, 14 + 13) == 1.0F);
		REQUIRE(feature(tensor, square, 28 + 13) == 0.0F);
		REQUIRE(feature(tensor, square, 118) == 1.0F);
		REQUIRE(feature(tensor, square, 116) == 0.0F); // raw e3 EP has no legal capture
	}
	throws<std::invalid_argument>([] { (void)encoding::orient_square(-1, Color::White); });
	throws<std::invalid_argument>([] { (void)encoding::orient_square(64, Color::Black); });
	throws<std::invalid_argument>([] { (void)encoding::orient_square(0, static_cast<Color>(2)); });
}

void historical_repetition_is_prefix_based()
{
	auto position = Position::start();
	knight_cycle(position);
	const auto after_one_cycle = encoding::encode_features(position);
	REQUIRE(feature(after_one_cycle, 0, 12) == 1.0F);
	REQUIRE(feature(after_one_cycle, 0, 14 + 12) == 0.0F);
	REQUIRE(feature(after_one_cycle, 0, 4 * 14 + 12) == 0.0F);
	REQUIRE(feature(after_one_cycle, 0, 4 * 14 + 13) == 1.0F);
	REQUIRE(feature(after_one_cycle, 0, 5 * 14 + 13) == 0.0F);
	knight_cycle(position);
	const auto full = encoding::encode_features(position);
	for (int square = 0; square < 64; ++square) {
		for (std::size_t frame = 0; frame < 8; ++frame) {
			REQUIRE(feature(full, square, frame * 14 + 13) == 1.0F);
			REQUIRE(feature(full, square, frame * 14 + 12) == (frame <= 4 ? 1.0F : 0.0F));
		}
	}
	// FEN import knows no earlier frames, even when clocks/board are identical.
	const auto imported = encoding::encode_features(Position::from_fen(position.fen()));
	REQUIRE(feature(imported, 0, 12) == 0.0F);
	REQUIRE(feature(imported, 0, 14 + 13) == 0.0F);
	const auto saved = full;
	position.play_uci("g1f3");
	position.unmake_move();
	REQUIRE(encoding::encode_features(position) == saved);
}

void current_rule_features()
{
	const auto position = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 75 1");
	const auto tensor = encoding::encode_features(position);
	for (int square = 0; square < 64; ++square) {
		REQUIRE(feature(tensor, square, 112) == 0.0F);
		REQUIRE(feature(tensor, square, 113) == 1.0F);
		REQUIRE(feature(tensor, square, 114) == 1.0F);
		REQUIRE(feature(tensor, square, 115) == 0.0F);
		REQUIRE(feature(tensor, square, 117) == 0.5F);
		REQUIRE(feature(tensor, square, 118) == 1.0F);
	}
	for (const auto fen : {"7k/8/8/8/8/8/8/KR6 w - - 150 1",
			       "7k/8/8/8/8/8/8/KR6 w - - 18446744073709551615 1"}) {
		const auto clamped = encoding::encode_features(Position::from_fen(fen));
		for (int square = 0; square < 64; ++square)
			REQUIRE(feature(clamped, square, 117) == 1.0F);
	}
	const auto white_ep = encoding::encode_features(
		Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"));
	const auto black_ep = encoding::encode_features(
		Position::from_fen("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"));
	for (int square = 0; square < 64; ++square) {
		REQUIRE(feature(white_ep, square, 116) == (square == 43 ? 1.0F : 0.0F));
		REQUIRE(feature(black_ep, square, 116) == (square == 43 ? 1.0F : 0.0F));
	}
	const auto pinned_ep = encoding::encode_features(
		Position::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1"));
	for (int square = 0; square < 64; ++square)
		REQUIRE(feature(pinned_ep, square, 116) == 0.0F);
}

void policy_roundtrips()
{
	const std::array fixtures = {
		Position::start(),
		Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
		Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
		Position::from_fen("1r2k3/P7/8/8/8/8/7p/4K1R1 w - - 0 1"),
		Position::from_fen("1r2k3/P7/8/8/8/8/7p/4K1R1 b - - 0 1"),
		Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
	};
	for (const auto &position : fixtures) {
		const auto moves = position.legal_moves();
		const auto indices = encoding::legal_indices(position);
		const auto mask = encoding::legal_mask(position);
		REQUIRE(indices.size() == moves.size());
		REQUIRE(std::set<std::size_t>(indices.begin(), indices.end()).size() == moves.size());
		REQUIRE(std::accumulate(mask.begin(), mask.end(), 0) == static_cast<int>(moves.size()));
		for (std::size_t i = 0; i < moves.size(); ++i) {
			REQUIRE(indices[i] < encoding::policy_size);
			REQUIRE(mask[indices[i]] == 1);
			REQUIRE(encoding::encode_move(position, moves[i]) == indices[i]);
			REQUIRE(encoding::decode_move(position, indices[i]) == moves[i]);
		}
	}
	const auto white = fixtures[3];
	const auto black = fixtures[4];
	REQUIRE(encoding::encode_move(white, Move::from_uci("a7a8q")) == 48 * 64 + 56);
	REQUIRE(encoding::encode_move(white, Move::from_uci("a7a8n")) == 4096);
	REQUIRE(encoding::encode_move(white, Move::from_uci("a7b8b")) == 4096 + 64 + 1);
	REQUIRE(encoding::encode_move(white, Move::from_uci("a7b8r")) == 4096 + 128 + 1);
	REQUIRE(encoding::encode_move(black, Move::from_uci("h2h1q")) == 55 * 64 + 63);
	REQUIRE(encoding::encode_move(black, Move::from_uci("h2h1n")) == 4096 + 63);
	REQUIRE(encoding::encode_move(black, Move::from_uci("h2g1b")) == 4096 + 64 + 62);
	REQUIRE(encoding::encode_move(black, Move::from_uci("h2g1r")) == 4096 + 128 + 62);
	const auto start = Position::start();
	REQUIRE(encoding::encode_move(start, Move::from_uci("e2e4")) == 12 * 64 + 28);
	throws<std::invalid_argument>([&] { (void)encoding::encode_move(start, Move::from_uci("e2e5")); });
	throws<std::invalid_argument>([&] { (void)encoding::encode_move(start, Move::from_uci("a2a4q")); });
	throws<std::invalid_argument>([&] { (void)encoding::encode_move(start, Move{}); });
	for (const auto index : {std::size_t{0}, std::size_t{4096}, encoding::policy_size,
				std::numeric_limits<std::size_t>::max()})
		throws<std::invalid_argument>([&] { (void)encoding::decode_move(start, index); });
}

void masking_and_softmax()
{
	const auto start = Position::start();
	const auto indices = encoding::legal_indices(start);
	std::array<float, encoding::policy_size> logits;
	logits.fill(std::numeric_limits<float>::quiet_NaN());
	for (const auto index : indices)
		logits[index] = 0.0F;
	const auto uniform = encoding::legal_softmax(start, std::span<const float>(logits));
	REQUIRE(uniform.size() == 20);
	for (double probability : uniform)
		REQUIRE(near(probability, 0.05));
	logits[indices[0]] = 1000.0F;
	logits[indices[1]] = 999.0F;
	const auto concentrated = encoding::legal_softmax(start, std::span<const float>(logits));
	REQUIRE(near(concentrated[0], 1.0 / (1.0 + std::exp(-1.0))));
	REQUIRE(near(concentrated[1], std::exp(-1.0) / (1.0 + std::exp(-1.0))));
	REQUIRE(near(std::accumulate(concentrated.begin(), concentrated.end(), 0.0), 1.0));
	logits[indices[0]] = std::numeric_limits<float>::infinity();
	throws<std::invalid_argument>([&] { (void)encoding::legal_softmax(start, std::span<const float>(logits)); });
	logits[indices[0]] = std::numeric_limits<float>::quiet_NaN();
	throws<std::invalid_argument>([&] { (void)encoding::legal_softmax(start, std::span<const float>(logits)); });
	throws<std::invalid_argument>([&] {
		(void)encoding::legal_softmax(start, std::span<const float>(logits).first(4287));
	});
	std::array<double, encoding::policy_size> doubles{};
	for (std::size_t i = 0; i < indices.size(); ++i)
		doubles[indices[i]] = static_cast<double>(i) - 10.0;
	const auto original = encoding::legal_softmax(start, std::span<const double>(doubles));
	for (auto &logit : doubles)
		logit += 1000000.0;
	const auto shifted = encoding::legal_softmax(start, std::span<const double>(doubles));
	for (std::size_t i = 0; i < original.size(); ++i)
		REQUIRE(near(original[i], shifted[i]));
	const auto mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
	REQUIRE(encoding::legal_indices(mate).empty());
	REQUIRE(encoding::legal_softmax(mate, std::span<const float>(logits)).empty());
	const auto empty_mask = encoding::legal_mask(mate);
	REQUIRE(std::accumulate(empty_mask.begin(), empty_mask.end(), 0) == 0);
	// Draw adjudication belongs to the caller; legal geometry remains present.
	const auto drawn = Position::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1");
	REQUIRE(drawn.outcome().is_draw());
	REQUIRE(!encoding::legal_indices(drawn).empty());
	REQUIRE(!encoding::legal_softmax(drawn, std::span<const double>(doubles)).empty());
}

void wdl_contract()
{
	REQUIRE(near(encoding::wdl_value({0.7, 0.2, 0.1}), 0.6));
	REQUIRE(encoding::wdl_value({1.0, 0.0, 0.0}) == 1.0);
	REQUIRE(encoding::wdl_value({0.0, 1.0, 0.0}) == 0.0);
	REQUIRE(encoding::wdl_value({0.0, 0.0, 1.0}) == -1.0);
	for (const encoding::Wdl wdl : {
		encoding::Wdl{-0.1, 0.6, 0.5}, encoding::Wdl{1.1, 0.0, -0.1},
		encoding::Wdl{0.0, 0.0, 0.0}, encoding::Wdl{0.4, 0.4, 0.4},
		encoding::Wdl{std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0},
		encoding::Wdl{0.0, std::numeric_limits<double>::infinity(), 0.0},
	})
		throws<std::invalid_argument>([&] { (void)encoding::wdl_value(wdl); });
}

} // namespace

int main()
{
	const std::vector<std::pair<std::string_view, std::function<void()>>> tests = {
		{"dimensions and padding", dimensions_and_padding},
		{"current perspective for every frame", current_perspective_for_every_frame},
		{"historical repetition prefixes", historical_repetition_is_prefix_based},
		{"current rule features", current_rule_features},
		{"policy roundtrips", policy_roundtrips},
		{"masking and softmax", masking_and_softmax},
		{"WDL contract", wdl_contract},
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
	return failed == 0 ? 0 : 1;
}
