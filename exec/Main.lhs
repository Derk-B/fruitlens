\section{Wrapping it up in an exectuable}\label{sec:Main}

We will now use the library from the previous sections in a program.

\begin{code}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import FruitLens

main :: IO ()
main = startServer 8080
\end{code}

We can run this program with the commands:

\begin{verbatim}
stack build
stack exec fruitlens-exe
\end{verbatim}
