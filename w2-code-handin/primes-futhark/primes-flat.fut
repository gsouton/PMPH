------------------------
--- Sgm Scan Helpers ---
------------------------

-- segmented scan with (+) on floats:
let sgmSumI64 [n] (flg : [n]bool) (arr : [n]i64) : [n]i64 =
  let flgs_vals = 
    scan ( \ (f1, x1) (f2,x2) -> 
            let f = f1 || f2 in
            if f2 then (f, x2)
            else (f, x1 + x2) )
         (false, 0i64) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

let mkFlagArray [m] (shape_array: [m]i64) (zero: i64)
                       (value_array: [m]i64): []i64 =
  let shape_rot = map(\i->if i == 0 then 0 else shape_array[i-1])(iota m)
  let shape_scan = scan (+) 0 shape_rot
  let len = shape_scan[m-1] + shape_array[m-1]
  let shape_index = map2 (\shp index ->
                           if shp == 0 then -1
                           else index
                          ) shape_array shape_scan
  in scatter (replicate len zero) shape_index value_array


let mkFlagArrayTuple [m] (shape_array: [m]i64) (zero: (i64,i64))
                       (value_array: [m](i64, i64)): [](i64, i64) =
  let shape_rot = map(\i->if i == 0 then 0 else shape_array[i-1])(iota m)
  let shape_scan = scan (+) 0 shape_rot
  let len = shape_scan[m-1] + shape_array[m-1]
  let shape_index = map2 (\shp index ->
                           if shp == 0 then -1
                           else index
                          ) shape_array shape_scan
  in scatter (replicate len zero) shape_index value_array


-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 } auto output

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knowns the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the right `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
      let not_primes = replicate flat_size 0
      
      let mm1s = map (\p -> (len/p) - 1) sq_primes
      let nn = length sq_primes
      let flag = map (\i -> if i == 0 then false else true) 
                     ((mkFlagArray mult_lens 0 sq_primes) :> [flat_size]i64)
      let vals = map (\f -> if f then 0 else 1) flag
      let iots = sgmSumI64 flag vals
      let arr = map (+2) iots
      --let flag_mm1 = mkFlagArray mm1s 0 mm1s
      --let flag_sqrn = mkFlagArray mm1s 0 sq_primes
      let tmp = mkFlagArrayTuple mm1s (0,0) (zip mult_lens sq_primes) :> [flat_size](i64,i64)
      let (flag_mm1, flag_sqrn) = unzip tmp
      let ps = sgmSumI64 (map (\i -> if i == 0 then false else true) flag_mm1) flag_sqrn
      let not_primes = map2 (*) ps arr

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
-- $ futhark opencl primes-flat.fut
-- $ echo "10000000" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
let main (n : i64) : []i64 = primesFlat n
