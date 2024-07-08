
/****************************************************************************
   (C) 2012 Doug Johnson

    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*****************************************************************************/

#include "des.h"
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define TEST_SIZE 102400000
int main(int argc, char** argv) {
    long num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    long test_size = TEST_SIZE;
    if (argc > 1)
       test_size = atol(argv[1]);
    if (argc > 2)
       num_threads = atol(argv[2]);
    #ifdef UNIT_TESTS
    return DESTestCases();
    #else
    uint64_t* output = malloc(test_size * sizeof(uint64_t));
	if (output == NULL) {
		fprintf(stderr, "Out of memory.\n");
	}
    #ifdef OPENCL
    DESBruteForceEncryptCL(output, test_size, 0xA000000000, 0xFFFFFFFF);
    #else
    DESBruteForceEncrypt(output, test_size, 0xA000000000, 0xFFFFFFFF, num_threads);
    #endif
    #endif
}
