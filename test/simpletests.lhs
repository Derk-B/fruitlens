\section{Simple Tests}
\label{sec:simpletests}

We now use the library QuickCheck to randomly generate input for our functions
and test some properties.

\begin{code}
module Main where

import FruitLens
import NeuralNetwork
import API
import Utils

import Test.Hspec
import Test.Hspec.QuickCheck
\end{code}

The following uses the HSpec library to define different tests.

\begin{code}
main :: IO ()
main = hspec $ do
  describe "FruitLens" $ do
    it "should have a placeholder test" $
      True `shouldBe` True
    -- Add more tests here as the project develops
\end{code}

To run the tests, use \verb|stack test|.

To also find out which part of your program is actually used for these tests,
run \verb|stack clean && stack test --coverage|. Then look for ``The coverage
report for ... is available at ... .html'' and open this file in your browser.
See also: \url{https://wiki.haskell.org/Haskell_program_coverage}.
