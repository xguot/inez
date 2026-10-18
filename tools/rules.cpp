// SPDX-License-Identifier: GPL-3.0-only
#include "inez/chess.hpp"

#include <algorithm>
#include <charconv>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string json_string(std::string_view value)
{
	constexpr char hex[] = "0123456789abcdef";
	std::string result = "\"";
	for (unsigned char ch : value) {
		if (ch == '"' || ch == '\\') {
			result += '\\';
			result += static_cast<char>(ch);
		} else if (ch < 0x20) {
			result += "\\u00";
			result += hex[ch >> 4];
			result += hex[ch & 0xf];
		} else {
			result += static_cast<char>(ch);
		}
	}
	return result + '"';
}

std::string_view trim(std::string_view value)
{
	const auto begin = value.find_first_not_of(" \t\r\n");
	if (begin == std::string_view::npos)
		return {};
	const auto end = value.find_last_not_of(" \t\r\n");
	return value.substr(begin, end - begin + 1);
}

const char *termination_name(inez::Termination termination)
{
	using enum inez::Termination;
	switch (termination) {
	case Ongoing: return "ongoing";
	case Checkmate: return "checkmate";
	case Stalemate: return "stalemate";
	case InsufficientMaterial: return "insufficient_material";
	case FiftyMoveRule: return "fifty_moves";
	case ThreefoldRepetition: return "threefold_repetition";
	case SeventyFiveMoveRule: return "seventyfive_moves";
	case FivefoldRepetition: return "fivefold_repetition";
	}
	throw std::logic_error("unknown termination");
}

std::string snapshot(const inez::Position &position)
{
	std::vector<std::string> moves;
	for (const auto &move : position.legal_moves())
		moves.push_back(move.uci());
	std::sort(moves.begin(), moves.end());

	const auto outcome = position.outcome();
	std::string result = "{\"fen\":" + json_string(position.fen());
	result += ",\"moves\":[";
	for (std::size_t i = 0; i < moves.size(); ++i) {
		if (i)
			result += ',';
		result += json_string(moves[i]);
	}
	result += "]";
	const auto flag = [&result](const char *name, bool value) {
		result += ",\"";
		result += name;
		result += value ? "\":true" : "\":false";
	};
	flag("check", position.in_check());
	flag("insufficient_material", position.is_insufficient_material());
	flag("can_claim_fifty", position.can_claim_fifty_moves());
	flag("can_claim_threefold", position.can_claim_threefold_repetition());
	result += ",\"repetition_count\":" + std::to_string(position.repetition_count());
	result += ",\"outcome\":" + json_string(termination_name(outcome.termination));
	result += ",\"winner\":";
	if (!outcome.winner)
		result += "null";
	else
		result += *outcome.winner == inez::Color::White ? "\"white\"" : "\"black\"";
	return result + '}';
}

void require_no_arguments(std::string_view arguments)
{
	if (!arguments.empty())
		throw std::invalid_argument("unexpected command arguments");
}

void usage(std::ostream &out)
{
	out << "Inez standard-chess rules inspector\n"
	       "Read commands from stdin; each response is one JSON line.\n"
	       "Commands: start, fen <FEN>, push <UCI move>, pop, state,\n"
	       "          perft <nonnegative depth>, quit.\n";
}

} // namespace

int main(int argc, char **argv)
{
	if (argc == 2 && std::string_view(argv[1]) == "--help") {
		usage(std::cout);
		return 0;
	}
	if (argc != 1) {
		usage(std::cerr);
		return 2;
	}

	inez::Position position;
	std::string line;
	while (std::getline(std::cin, line)) {
		try {
			const auto input = trim(line);
			const auto end = input.find_first_of(" \t");
			const auto command = input.substr(0, end);
			const auto arguments = end == std::string_view::npos
				? std::string_view{} : trim(input.substr(end + 1));
			if (command == "quit") {
				require_no_arguments(arguments);
				break;
			}
			if (command == "perft") {
				int depth = -1;
				if (arguments.empty())
					throw std::invalid_argument("perft requires a depth");
				const auto parsed = std::from_chars(arguments.data(),
					arguments.data() + arguments.size(), depth);
				if (parsed.ec != std::errc{} ||
				    parsed.ptr != arguments.data() + arguments.size() || depth < 0)
					throw std::invalid_argument("invalid perft depth");
				const auto nodes = position.perft(depth);
				std::cout << "{\"nodes\":" << nodes << "}\n";
			} else {
				if (command == "fen") {
					position = inez::Position::from_fen(arguments);
				} else if (command == "push") {
					position.play_uci(arguments);
				} else if (command == "start") {
					require_no_arguments(arguments);
					position = inez::Position::start();
				} else if (command == "pop") {
					require_no_arguments(arguments);
					position.unmake_move();
				} else if (command == "state") {
					require_no_arguments(arguments);
				} else {
					throw std::invalid_argument("unknown command");
				}
				std::cout << snapshot(position) << '\n';
			}
		} catch (const std::exception &error) {
			std::cout << "{\"error\":" << json_string(error.what()) << "}\n";
		}
		std::cout.flush();
	}
	return std::cin.bad() || !std::cout ? 1 : 0;
}
