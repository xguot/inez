// SPDX-License-Identifier: GPL-3.0-only
#include "inez/encoding.hpp"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

void require_no_arguments(std::string_view arguments)
{
	if (!arguments.empty())
		throw std::invalid_argument("unexpected command arguments");
}

std::string snapshot(const inez::Position &position)
{
	namespace encoding = inez::encoding;
	const auto features = encoding::encode_features(position);
	const auto legal = position.legal_moves();
	const auto indices = encoding::legal_indices(position);
	std::vector<std::pair<std::string, std::size_t>> moves;
	moves.reserve(legal.size());
	for (std::size_t i = 0; i < legal.size(); ++i)
		moves.emplace_back(legal[i].uci(), indices[i]);
	std::sort(moves.begin(), moves.end());

	std::ostringstream output;
	output << std::setprecision(std::numeric_limits<float>::max_digits10)
	       << "{\"contract_version\":" << encoding::contract_version
	       << ",\"fen\":" << json_string(position.fen()) << ",\"features\":[";
	for (std::size_t i = 0; i < features.size(); ++i) {
		if (i != 0)
			output << ',';
		output << features[i];
	}
	output << "],\"legal_moves\":[";
	for (std::size_t i = 0; i < moves.size(); ++i) {
		if (i != 0)
			output << ',';
		output << json_string(moves[i].first);
	}
	output << "],\"legal_indices\":[";
	for (std::size_t i = 0; i < moves.size(); ++i) {
		if (i != 0)
			output << ',';
		output << moves[i].second;
	}
	output << "]}";
	return output.str();
}

void usage(std::ostream &out)
{
	out << "Inez versioned neural encoding inspector\n"
	       "Read commands from stdin; each response is one JSON line.\n"
	       "Commands: start, fen <FEN>, push <UCI move>, pop, encode, quit.\n";
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
			} else if (command == "encode") {
				require_no_arguments(arguments);
			} else {
				throw std::invalid_argument("unknown command");
			}
			// Build a complete response before writing, so errors never emit
			// a partial JSON object on stdout.
			std::cout << snapshot(position) << '\n';
		} catch (const std::exception &error) {
			std::cout << "{\"error\":" << json_string(error.what()) << "}\n";
		}
		std::cout.flush();
	}
	return std::cin.bad() || !std::cout ? 1 : 0;
}
