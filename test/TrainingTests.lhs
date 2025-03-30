\section{Tests}\label{sec:tests}

We use the QuickCheck library to randomly generate input data for testing our neural network implementation.
The main goal is to verify that our network can handle arbitrary input data and produce valid predictions
after training. 

Due to the nature of neural networks and how the training works, it's impossible to ship actual training images with the source code to perform the test. Therefore, we have decided to randomly generate inputs, and as long as the training does not throw errors and the inference part yields a result, the tests are considered passed. 

\subsection{Test Structure}
The test suite consists of several components:

\begin{enumerate}
  \item Random data generators for creating test inputs
  \item A helper function to encapsulate the training and prediction process
  \item Property tests that verify the network's behavior
\end{enumerate}

\subsection{Data Generators}
We define several QuickCheck generators to create random test data:

\begin{itemize}
  \item \texttt{genNormalizedFloat}: Generates random float values between 0 and 1, used for pixel values
  \item \texttt{genPixel}: Creates a random RGB pixel as a list of 3 normalized floats
  \item \texttt{genImage}: Builds a $100\times100$ RGB image as a 3D array of floats
  \item \texttt{genFruitLabel}: Produces a random one-hot encoded label for either Apple [1.0, 0.0] or Banana [0.0, 1.0]
  \item \texttt{genTrainingExample}: Combines random images and labels into training pairs
\end{itemize}

\subsection{Test Cases}
The main test verifies that:
\begin{itemize}
  \item The network can be trained on randomly generated data
  \item The training process completes without errors
  \item The trained model produces valid fruit type predictions (either Apple or Banana)
\end{itemize}

\hide {
\begin{code}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Main where

import FruitLens

import Test.Hspec ( hspec, describe, it, shouldReturn )
import Test.QuickCheck
import Control.Monad (replicateM)
\end{code}
}

\begin{code}

-- Generate a random float between 0 and 1
genNormalizedFloat :: Gen Float
genNormalizedFloat = choose (0, 1)

-- Generate a random RGB pixel (3 normalized floats)
genPixel :: Gen [Float]
genPixel = replicateM 3 genNormalizedFloat

-- Generate a random 100x100 image with RGB channels
genImage :: Gen [[[Float]]]
genImage = do
  -- Generate 100 rows
  rows <- replicateM 100 $ do
    -- Each row has 100 pixels
    replicateM 100 genPixel

  return rows

-- Generate a random fruit label (one-hot encoded)
genFruitLabel :: Gen [Float]
genFruitLabel = do
  isApple <- arbitrary
  return $ if isApple then [1.0, 0.0] else [0.0, 1.0]

-- Generate a random training example (image and label pair)
genTrainingExample :: Gen (Image, [Float])
genTrainingExample = do
  img <- genImage
  label <- genFruitLabel
  return (img, label)

-- Helper function to run the training process
trainAndPredict :: [(Image, [Float])] -> IO Bool
trainAndPredict trainingData = do
  initialModel <- newModelFC
  trainedModel <- trainModel initialModel trainingData 1 0.01
  let predictions = map (\(img, _) -> predictFruit trainedModel img) trainingData
  return $ all (\p -> p `elem` [Apple, Banana]) predictions

main :: IO ()
main = hspec $ do
  describe "FruitLens" $ do
    it "should train on randomly generated data without crashing" $ do
      -- Generate 5 random training examples using QuickCheck
      trainingData <- generate $ replicateM 5 genTrainingExample
      -- Run the training and prediction, should return True
      trainAndPredict trainingData `shouldReturn` True

\end{code}

To run the tests, use \verb|stack test|.

