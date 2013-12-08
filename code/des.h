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

#include <inttypes.h>

uint64_t DESExpandParity(uint64_t key);

uint64_t DESEncrypt(uint64_t input, uint64_t key);

uint64_t DESDecrypt(uint64_t input, uint64_t key);

int DESBruteForceEncrypt(uint64_t* output, 
                          uint32_t size, 
                          uint64_t key_start,
                          uint64_t data, 
                          uint32_t num_threads);

int DESBruteForceEncrypt(uint64_t* output, 
                          uint32_t size, 
                          uint64_t key_start,
                          uint64_t data, 
                          uint32_t num_threads);


int DESBruteForceEncryptCL(uint64_t* output, 
                          uint32_t size, 
                          uint64_t key_start,  
                          uint64_t data);

int DESBruteForceDecryptCL(uint64_t* output, 
                          uint32_t size, 
                          uint64_t key_start,  
                          uint64_t data);

int DESTestCases();
