\section{FruitLens Core Module}\label{sec:FruitLens}

This is the main module that re-exports all the functionality from the FruitLens submodules.

\begin{code}
module FruitLens
  ( module API
  , module NeuralNetwork
  , module Utils
  ) where

import API
import NeuralNetwork
import Utils
\end{code}
