# PMPH assignement 4

## Task 1



### 1.a

MSI | Requests   | Time to carry out protocol action   | Traffic (bytest)  |
|----------------|-------------------------------------|-------------------| -------------- |
|                | R1/X                                | 40(read-exc)      | (6+32)         |
|                | W1/X                                | 10(Bus-up)        | 10             |
|                | W1/X                                | 1(write-hit)      | NA             |
|                | R2/X                                | 40(readReg)       | (6+32)         |

For MSI the total number of cycles would be evaluated to (40+10)*4 = 200 cycles
and the traffic evaluated to (38 + 10) * 4 = 192 bytes



MESI | Requests   | Time to carry out protocol action   | Traffic (bytest) |
|----------------|-------------------------------------|-------------------| -------------- |
|                | R1/X                                | 40(read-exc)      | (6+32)         |
|                | W1/X                                | 1(write-hit)      | NA              |
|                | W1/X                                | 1(write-hit)      | NA             |
|                | R2/X                                | 40(readReg)       | (6+32)         |


## Task 2

- 1. cold miss
- 2. cold miss
- 3. cold miss
- 4. hit
- 5. cold miss
- 6. false sharing
- 7. hit
- 8. replacement hit
- 9. true sharing


## Task 3


3.a. Handling read cache miss with home node == local node and memory copy is clean
 - Cycles: It takes 50 cycles (directory lookup) 
 - Traffic: No traffic is needed since home == local.


3.b. Handling read cache miss with home node == local node and memory copy is dirty 
 - Cycles: (directory lookup) 50 cycles + 
 (RemRd) 20 cycles + (lookup cache dirty) 50 + (flush) 100 + (mem-update) 50 
 = 270 cycles
- Traffic: 6 (remRd) + (6+32) (flush) = 44 bytes

3.c. Handling read cache miss with home != local and memory copy is clean 
- Cycles: (busRead) 20 + (lookup) 50 + (flush) 100 + (installing in cache local) 50
= 220
- Traffic: 6 (busRead) + (6+32) (flush) = 44 bytes

3.d. Handling read cache miss with home != local, home = remote, and memory copy is dirty 
- Cycles: (busRead) 20 + (dir lookup) 50 + (flush) 100 + (install in cache) 50
= 220
- Traffic: 6 (busRead) + (6+32) (flush) = 44 bytes

3.e. Handling read cache miss with home != local, home != remote, and memory copy is dirty 
- Cycles: (busRead) 20 + (dir lookup) 50 + (remRd) 20 + (cache) 50 + (flush) 100 + (updatemem) 50 + (flush) 100 + ( install in cache) 50 (flush) 100 + (install in cache) 50
 \> 600
- Traffic: 6 (busRead) + 6(remRd) + 2 * (6+32) = 88 bytes

DASH: a,b,c,d <=> MSI

(e) (BusRd) 20 + (lookup) 50 + (remRd) 20 + (lookup cache) 50 + (flush) 100 + (cache update) 50
traffic: 6 + 6 + 2*(6+32)

## Task 4

4.a network diameter: n = 16

4.b bisection bandwidth: bisection width * line-bdanwidth = 32 * 100 Mbit/s = 3.2 Gbit/s

4.c bandwidth node: (#links * line-bandwidth)/(#nodes) = 512 * 100/ 256 = 200 Mbit/sec

#### Notes
There 2n^2 links in a n by n tori


## Task 5

5.a bisecion of hypercube = N/2,  bisection of tori = 2*sqrt(N). 

- N/2 > 2*sqrt(N)
- N/sqrt(N) > 4
- N > 16

5.b network diameter tori: n, network of hypercube: log(n)
- log_2(n) > n

Network diameter

Tori -> N = 64, n = 8, diameter: 8
Hyp  -> N = 64, n = 64/2 = 32, diameter: log(64)






