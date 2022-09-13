-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [0]
-- }
-- output {
--    1
-- }
--
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
--
-- compiled input {
--    [1, -2, -2, 42, 42, 42, 42, 42, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
--
-- compiled input {
--    [0, 0, 1, 2, 3, 4]
-- }
-- output {
--    2
-- }
--
-- compiled input {
--    [4, 5, 1, 2, 6, 6]
-- }
-- output {
--    2
-- }
--
-- compiled input {
--    [1, -2, 2, 2, 2, 42, 2, 42, 2, 2, -2, 2, 2]
-- }
-- output {
--    3
-- }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
