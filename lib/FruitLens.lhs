\section{FruitLens Core Module}\label{sec:FruitLens}

This is the main module that re-exports all the functionality from the FruitLens submodules.

\begin{code}
module FruitLens
  ( module AI,
    module API,
    module Utils,
  )
where

import AI
import API
import Utils
\end{code}
