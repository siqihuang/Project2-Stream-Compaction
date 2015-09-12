#pragma once

namespace StreamCompaction {
namespace Radix {
    void scan(int n, int *odata, const int *idata);
	void radix(int n, int *odata, const int *idata);
}
}