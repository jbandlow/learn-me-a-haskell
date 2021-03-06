import qualified Data.List as List
import qualified Data.Map as Map

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
--  helloString !! 1 == 'e'
--   Indexing outside of the list is an error.
-- <, >, <=, >= compare lists in lex order.
--Common list functions:
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
numUniques = length . List.nub

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
sum'' = List.foldl (+) 0

-- Reimplementing reverse, a more clever illustration:
reverse' :: [a] -> [a]
reverse' = List.foldl (\acc x -> x : acc) []
-- There are also fold?1 functions, which take the first list value as the
-- starting_value. These don't work on empty lists.
--
-- Scans are like folds, but return a list of all intermediate results.
--      scanl (+) 0 [3, 2, 1] == [0, 3, 5, 6]

-- TODO: Talk about Maybe

-- We can create our own types.  Ex:
data Point = Point Float Float deriving (Show)
data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

-- On the left side of a "data" declaration is the name of the type we are
-- defining (and possibly type parameters, which I'll discuss later). On
-- the right side, we define "data constructors" for this type. We can think of
-- these as functions with arguments, but nothing is done with the arguments,
-- we just hold on to them.  The pipe "|" separates these data constructors and
-- "deriving (typeclass)" asks Haskell to automatically write the functions
-- necessary for this type to belong to that typeclass.

-- Functions look like we'd expect.
perimeter :: Shape -> Float
perimeter (Circle _ r) = 2 * pi * r
perimeter (Rectangle (Point x1 y1) (Point x2 y2)) = 2 * ((abs $ x1 - x2) + (abs $ y1 - y2))

-- We can also define types using Record syntax:
data Car = Car { company :: String
               , model :: String
               , year :: Int
               } deriving (Show)
-- This defines accessor functions automatically:
--    mygt = Car "Ford" "Mustang" 1997
--    year mygt == 1997

-- We can also define types with type parameters, which allow for type-flexibility
-- in the data constructors.
data Vector a = Vector a a a deriving (Show)
-- The 'a' here can be any type, we will probably only use it with numeric types.
vplus :: (Num t) => Vector t -> Vector t -> Vector t
(Vector x y z) `vplus` (Vector u v w) = Vector (x + u) (y + v) (z + w)

vsize :: (Floating t) => Vector t -> t
vsize (Vector x y z) = (x**2 + y**2 + z**2) ** (1 / 3)

scalarLMultiply :: (Num t) => t -> Vector t -> Vector t
m `scalarLMultiply` (Vector x y z) = Vector (m*x) (m*y) (m*z)
-- v = Vector 1.0 1.0 1.0
-- vsize v == 3 ** (1 / 3)
-- 2 `scalarLMultiply` v == v `vplus` v FAILS b/c we haven't defined ==.
-- If we derive from Eq, it would work.  Common typeclass to derive from:
--   Eq, Ord, Enum, Bounded, Show, Read

-- We can also define a type to be a synonym for an existing type. Eg:
type PhoneNumber = String
type Name = String
type PhoneBook = [(Name,PhoneNumber)]

-- These can be parameterized as well:
type IntMap v = Map.Map Int v
-- Although we could have done "type IntMap = Map.Map Int" as well.

-- We can define infix functions composed only of symbols:
infixr 5 .*
(.*) :: (Num t) => t -> Vector t -> Vector t
a .* (Vector x y z) = Vector (a*x) (a*y) (a*z)

infixr 5 *.
(*.) :: (Num t) => Vector t -> t -> Vector t
(Vector x y z) *. a = Vector (x*a) (y*a) (z*a)


-- The next level of abstraction is to show how we can define typeclasses, and
-- implement types of that typeclass.
--
-- class (typeclass restrictions) => NewTypeclassName classVar where
--   Required function signatures, possibly with default implementations.
class (Show a) => Howdy a where
  howdy :: a -> String
  howdy x = "Howdy, " ++ show x

-- Define a class as usual.
data Friends = Jessica | Jason deriving (Show)

-- instance NewTypeclassName Class where
--   implementation of the required functions.
instance Howdy Friends where
  howdy x = "Hi, " ++ show x

-- In this case, there is a default implementation, so we don't need one.
data Johns = JohnI | JohnII deriving (Show)
instance Howdy Johns

-- howdy Jason == "Hi, Jason"
-- howdy JohnI == "Howdy, JohnI"

-- Nice example of implementing "make everything work as a Bool":
class YesNo a where
  yesno :: a -> Bool

instance YesNo Int where
  yesno 0 = False
  yesno _ = True

instance YesNo [a] where  -- Works for lists and strings
  yesno [] = False
  yesno _ = True

instance YesNo Bool where
  yesno = id

instance YesNo (Maybe a) where
  yesno Nothing = False
  yesno (Just _) = True

-- Functor is a typeclass that applies to parameterized types (like Vector,
-- specifically they must have kind: * -> *). The required function is fmap
-- and it should tell you how to combine a function (f: a->b) and an object
-- (v \in Vector $a) to produce an object (v \in Vector b). That is, if f
-- is a functor (with concrete types (f a), (f b), etc.) then the signature
-- of fmap is:
--   (a -> b) -> f a -> f b
--
-- Applicative functors are similar.  If f is a functor (so that you have
-- concrete types f Integer or f String, for example), then the key for
-- applicative functors is concrete types where the "inner type" is a function.
-- So f (Integer -> String), for example.  The required function is written
-- as <*> and the point is that you should be able to apply a "functored
-- function" to a "functored object".  So if F is an element of f (a -> b) and
-- A is an element of f a, we should be able to apply F to A to get B in f b.
-- This application is denoted <*> and has signature
--      f (a -> b) -> f a -> f b
-- For example, List is an applicative functor, so we can do:
--     [(+)] <*> [3, 4] <*> [1, 2, 3] == [4, 5, 6, 5, 6, 7]
-- Notice that the first application gives a list of curried functions, and
-- the second application gives the values.
--
-- There is also a shorthand notation where we can put the function in an
-- applicative context automatically:
--    (+) <$> [3, 4] <*> [1, 2, 3] == [4, 5, 6, 5, 6, 7]
--
-- Finally, there are monads.  A monad is an applicative functor, with one
-- more trick it can do.  Notice that the "special function" for functors
-- involve functions a -> b, and the "special function" for applicatives
-- involve values in f (a -> b) (which we can think of as functions). The
-- "special function" for monads involves function a -> m b. The point being
-- if you have a function that takes an ordinary value into a monadic context,
-- you should also be able to apply that function in a monadic context. The
-- special function is pronounced 'bind', is written as >>=, and has signature:
--    m a -> (a -> m b) -> m b
-- Note the strange order of the signature.  We use this infix like:
--   {monadic value}  >>= {monadic function}
-- but more commonly we don't write it at all, due to syntactic sugar I will
-- cover later.
