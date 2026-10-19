// SPDX-License-Identifier: GPL-3.0-only
#include "inez/uci.hpp"

#include <iostream>

int main()
{
	inez::Engine engine(std::cin, std::cout, std::cerr);
	engine.run();
	return 0;
}
