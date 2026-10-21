// SPDX-License-Identifier: GPL-3.0-only
#include "inez/uci.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#ifndef INEZ_ROOT_DIR
#define INEZ_ROOT_DIR "."
#endif

namespace {

bool is_executable(const std::string& path)
{
	return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

bool use_baseline_engine(int argc, char* argv[])
{
	if (argc > 1 && std::string_view(argv[1]) == "--baseline")
		return true;
	const char* baseline_env = std::getenv("INEZ_BASELINE");
	return baseline_env != nullptr && std::string_view(baseline_env) == "1";
}

} // namespace

int main(int argc, char* argv[])
{
	if (use_baseline_engine(argc, argv)) {
		inez::Engine engine(std::cin, std::cout, std::cerr);
		engine.run();
		return 0;
	}

	const std::string root_dir = INEZ_ROOT_DIR;
	std::string python_bin;
	bool search_path = false;

	if (const char* env_py = std::getenv("INEZ_PYTHON");
	    env_py != nullptr && env_py[0] != '\0') {
		python_bin = env_py;
		if (!is_executable(python_bin)) {
			std::cerr << "Inez: set INEZ_PYTHON to Python with training/requirements.txt installed.\n";
			return 1;
		}
	} else if (is_executable(root_dir + "/.venv-model/bin/python")) {
		python_bin = root_dir + "/.venv-model/bin/python";
	} else if (is_executable(root_dir + "/.venv/bin/python")) {
		python_bin = root_dir + "/.venv/bin/python";
	} else {
		python_bin = "python3";
		search_path = true;
	}

	::setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
	::setenv("PYTHONUNBUFFERED", "1", 1);

	std::string flag_u = "-u";
	std::string script = root_dir + "/tools/neural_uci.py";
	std::vector<char*> args;
	args.reserve(static_cast<std::size_t>(argc) + 3);
	args.push_back(python_bin.data());
	args.push_back(flag_u.data());
	args.push_back(script.data());
	for (int i = 1; i < argc; ++i)
		args.push_back(argv[i]);
	args.push_back(nullptr);

	if (search_path)
		::execvp(python_bin.c_str(), args.data());
	else
		::execv(python_bin.c_str(), args.data());

	std::cerr << "Inez: set INEZ_PYTHON to Python with training/requirements.txt installed.\n";
	return 1;
}
