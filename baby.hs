import Data.List  

-- ghci gives the interactive prompt, from the command line.
-- A file (like this) can be loaded with :l <filename>
-- The prompt can be set with ':set prompt "ghci> "'
-- Modules can be imported with ':m + Data.List'
--
-- Modules are here:
--      https://downloads.haskell.org/~ghc/latest/docs/html/libraries/
-- Hoogle is how to search for Haskell things:
--      https://www.haskell.org/hoogle/ 

-- ARITHMETIC
-- arithmetic works, with caveats
--   some conversions are done: 3 / 2 = 1.5
--   5 * -3 must be written as 5 * (-3)

-- == tests equality and /= is inequality

-- FUNCTIONS
-- Function application is with a space, not parens. 
-- Function application has the highest precedence in Haskell.
-- Functions with two args can be written infix with backticks:
--      10 `div` 2 == 5
-- A function definition: 
doubleMe x = x + x
--   doubleMe 5 == 10

-- More function examples
doubleUs x y = doubleMe x + doubleMe y 
doubleSmallNumber x = if x > 100
        then x
        else x * 2
-- ' is valid in function names, and often denotes a small change
doubleSmallNumber' x = (if x > 100 then x else x*2) + 1  

-- Functions CANNOT begin with an uppercase letter.
conanO'Brien = "It's a-me, Conan O'Brien!" 
-- Inside ghci, use "let" for assignments:
--    ghci> let conanO'Brien = "It's a-me, Conan O'Brien!" 

-- LISTS
-- All elements of a list must have the same type
lostNumbers = [4,8,15,16,23,42]
-- A string (defined with ") is just syntactic sugar for a list of characters
-- (defined with ').
helloString = "Hello"
--   helloString == ['H', 'e', 'l', 'l', 'o']

-- ++ concatenates lists
--   helloString ++ " world" == "Hello world"
--   ++ is linear-time in the length of the first list.
-- : appends a single element to the front of a list
--   0:lostNumbers == [0,4,8,15,16,23,42]
--   : is constant-time.
-- !! is the (0-based) index operator
-  helloString !! 1 == 'e'
--   Indexing outside of the list is an error.
-- <, >, <=, >= compare lists in lex order.
-Common list functions:
--   head,tail,init,last,length,null,reverse,take,drop,maximum,minimum
--   elem,sum,product,cycle,repeat,replicate
--
-- Ranges are given by ..
--   [1..5] == [1,2,3,4,5]
--   [2,4..11] == [2,4,6,8,10]
--   [10,9..6] == [10,9,8,7,6]  ([10..6] does NOT work)
-- 
-- Infinite lists work, and are lazy.
--   naturalNumbers = [0..]
--   take 4 naturalNumbers = [0,1,2,3]
--
-- List comprehensions work.
--   [ x | x <- [0..10], x `mod` 3 == 0 ] == [0, 3, 6, 9]
--   [x * y | x <- [1,2,3], y <- [4, 5] ] == [4, 5, 8, 10, 12, 15]
--   [ a + b | (a, b) <- [(1,2), (5,6), (0,0)] ] == [3, 11, 0]


-- TUPLES
-- Tuples are denoted by parens, and can contain heterogeneous data types.
-- But two tuples are the same type iff they have the same type in every
-- position.
-- Useful tuple functions:
--  fst,snd,zip


-- TYPES
-- Types can be examined in ghci with :t <object>
-- Common types:
--   Bool,Char,Int,Integer,Float,Double,String
--     Int is a machine length int, and Integer is arbitrary length
--     List types are denoted with brackets, eg: [Int]
--     String is an alias for [Char]
-- Functions have types <input type> -> <output type>. Every function in
--   Haskell is considered as having only one input, and possibly producing
--   another function as output.  It is customary to define the signature of a
--   function immediately before defining it:
addThree :: Int -> Int -> Int -> Int
addThree x y z = x + y + z
-- Some functions can operate on multiple types, so their signatures use
-- type-variables: :t head
--   head :: [a] -> a
-- Other functions can only operate on sum restricted set of types.  These
-- sets of types are called typeclasses. They look like this: :t elem
--   elem :: Eq a => a -> [a] -> Bool
-- which says that elem takes an element of some type, and a list of that
-- type, and returns a Bool. It can do this as long as the type is a member of
-- the typeclass Eq. The => is called a class constraint.
--
-- Common Typeclasses:
--   Eq: Types that support equality testing.  Functions, for example, do not.
--   Ord: Types that support ordering.
--   Show: Types that have a string representation
--   Read: Types that support the 'read' function which converts a string to
--     an object of the type, as long as Haskell can infer that type. Ex:
--       read "5" == 5
--     however, read "5" by itself is an error, because there is no context to
--     infer the type. We can provide the type with: read "5" :: Float
--     This is called a type annotation.
--   Enum: Types where elements have defined successors (succ) and
--     predecessors (pred).  Includes (), Bool, Char, Ordering, Int, Integer,
--     Float and Double.
--   Bounded: Types with a minimum (minBound) and maximum (maxBound).
--   Num: Types which support arithmetic.
--   Integral: Contains Int and Integer. fromIntegral will return an element
--     of a Num class from an Integral input. 
--   Floating: Contains Float and Double


-- MORE FUNCTIONS
-- Functions are defined by a type signature, followed by a collection of
-- definitions.  The first definition that matches the input is used. Good
-- practice is to include a catch-all at the end.
isSeven :: (Integral a, Show a) => a -> String
isSeven 7 = "Yes, that's a seven."
isSeven x = "No, " ++ show x ++ " is not seven."

-- There is a simple syntax for cases. These are called guards. When using
-- guards we use '=' for each guard, not after the parameter pattern.
factorial :: (Integral a) => a -> a
factorial n
    | n < 0     = 0  -- No it's not, but whatever. 
    | n == 0    = 1
    | otherwise = n * factorial (n - 1)

-- We can pattern match against tuples:
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)  
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

-- Matching the first item in a list is a common pattern, typically with
-- recursion.
sum' :: (Num a) => [a] -> a  
sum' [] = 0  
sum' (x:xs) = x + sum' xs  

-- We can also give a name to the whole pattern, by using a name followed by
-- the symbol @. This is called an "as pattern". 
capital :: String -> String  
capital "" = "Empty string, whoops!"  
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x] 

-- A 'where' clause can be used to define values that are used multiple times,
-- or for syntactic clarity. The scope is the current function definition
-- i.e., everything after <function name> <parameter pattern>, including all
-- guards.
circleSize :: (RealFloat a) => a -> String
circleSize r 
    | area > big    = "That's a big circle."
    | area > little = "That's a normal circle."
    | otherwise     = "That's a wee circle."
    where area = pi * r^2
          (little, big) = (5, 10)

-- More tightly scoped temporaries can be defined with "let" .. "in". These
-- only exist in the context of the current statement, and can no be used to
-- span guards. 
cylinderArea :: (RealFloat a) => a -> a -> a
cylinderArea r h =
    let sideArea = 2 * pi * r * h
        topArea  = pi * r^2
    in sideArea + 2 * topArea

-- "where" is only valid in function definitions, "let..in" is valid in any
-- statement:
--     4 * (let a = 9 in a + 1) + 2  == 42

-- in can be omitted, if there is already a context for the let.
calcBmis :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2]  

-- CASE statements.  They look like "case <item> of (<pattern> -> <result>)"
describeList :: [a] -> String  
describeList xs = "The list is " ++ case xs of [] -> "empty."  
                                               [x] -> "a singleton list."   
                                               xs -> "a longer list." 


-- RECURSION
-- This is a little bit magic.  This is slow:
myMax :: (Ord a) => [a] -> a
myMax [] = error "Empty list"
myMax [x] = x
myMax (x:xs) = if x > myMax xs then x else myMax xs

-- But this is fast:
myMax' :: (Ord a) => [a] -> a
myMax' [] = error "Empty list"
myMax' [x] = x
myMax' (x:xs)
    | x > tailMax   = x
    | otherwise     = tailMax
    where tailMax = myMax' xs

-- Some fun examples:
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
    let smaller = [y | y <- xs, y <= x]
        greater = [y | y <- xs, y > x]
    in quicksort smaller ++ [x] ++ quicksort greater

-- This is fast and easy to write, and fun to play with.
collatzSequence :: (Integral a) => a -> [a]
collatzSequence n
    | n <= 0    = [] 
    | n == 1    = n : []
    | even n    = n : collatzSequence (n `div` 2)
    | odd n     = n : collatzSequence (3 * n + 1)

-- Function application
-- infix functions can be paritally applied using sections.  Basically, apply
-- one parameter, and wrap it in parentheses, and Haskell will know how to
-- apply it.
--    map (`mod` 2) [1..10] == [1,0,1,0,1,0,1,0,1,0]
--    map (10 `mod`) [1..10] == [0,0,1,2,0,4,3,2,1,0]
--    map (/10) [5 * x | x <- [1..4]] == [0.5,1.0,1.5,2.0]


-- Count uniques by taking the length of the list with duplicates removed.
-- nub is the built in function that removes duplicates.
numUniques :: (Eq a) => [a] -> Int
numUniques = length . nub

-- takeWhile iterates through a possibly inifinite list, and stops when a
-- predicate returns false.
-- The sum of the odd squares which are less than 1000 is
--      sum (takeWhile (<1000) (filter odd (map (^2) [1..])))

-- $ is syntactic sugar for function application.  It's low precedence, so it
-- allows you to omit parentheses. Instead of 
--      sum (takeWhile (<1000) (filter odd (map (^2) [1..])))
--  we can write
--      sum $ takeWhile (<1000) $ filter odd $ map (^2) [1..]
--  which is more readable.

-- . is syntactic sugar for function composition. Instead of, e.g.,
--      negateAll = map (\x -> negate (abs x))
--  we can write
--      negateAll = map (negate . abs)

-- Summary of function notation:
--      negate sqrt 5       -- Error.
--      negate (sqrt 5)     -- This works.
--      negate $ sqrt 5     -- This works. 
--      (negate . sqrt) 5   -- This works.
--      negate . sqrt $ 5   -- This works.
--      (negate sqrt) 5     -- Error.
--      negate . sqrt 5     -- Error.      

-- lambda's are defined by \
--      map (\(x, y) -> x + y) [(1, 2), (3, 4)] == [3, 7]
-- That lambda took one argument which was an ordered pair. They can also take
-- multiple arguments:
--      (\x y z -> x + y + z) 1 2 3 == 6

-- Folds are used to process a list element by element.  foldl and foldr start
-- from the left (or right) side, and are of the form: 
--     foldl function starting_value list
-- To redefine the sum function:
sum' = foldl (+) 0 

-- Reimplementing reverse, a more clever illustration:
reverse' :: [a] -> [a]
reverse' = foldl (\acc x -> x : acc) []
-- There are also fold?1 functions, which take the first list value as the
-- starting_value. These don't work on empty lists.
--
-- Scans are like folds, but return a list of all intermediate results.
--      scanl (+) 0 [3, 2, 1] == [0, 3, 5, 6]

