---
title: My Experience with GO
date: 2024-05-15
categories: GO
---

My entire Professional background consists of roles focused on Machine Learning so, as expected, I have had to code almost exclusively in Python, with the occasional Matlab or Julia from time to time. Academically, however, I have my roots planted in Embedded Programming (due to my Bachelor's in Electronic Engineering) and Scientific Programming (due to my Master's in Mathematical Engineering), which is why I tend to favour design patterns that minimize time-complexity and can be easily parallelized. As a result, I have had to familiarize myself with the different abstraction levels of the C family, from bit-mask registry manipulation for individual models of the AVR family to convoluted Generic classes used to handle multiple numerical solvers for PDE libraries.

Despite its ever-presence in my career, I have never enjoyed programming in C or C++ as much as I enjoy working with Python. I can live with the lack of memory safety, and I must admit it's quite satisfying to watch the compilation bar go Brrr. Nevertheless, it's the lack of language *consistency* that truly irks me every time I try to work with them.

With C, every decently-sized project is a compilation of macro hacks and deeply-nested pointers, the documentation for which should not be expected. With C++, the 4-5 ways to perform the same operation incorporated across the different generations of the standard library, turns every sufficiently long-lived codebase into a family tree of programming paradigms. And for both of them: the lack of a standard package management is a feature that modern languages simply cannot afford to neglect.

It might come across as pedantic, but I do believe that developers being opinionated about their Software is a net positive, especially in the case of Programming Language design. Sure, in Software there is no single best way to solve a problem, and every solution lies on a spectrum (or as [CodeAesthetic](https://www.youtube.com/@CodeAesthetic) expertly put it, a [Triangle](https://youtu.be/tKbV6BpH-C8?si=kxM9dwkI5zzXeYNC)) of trade-offs. However, there should be guidelines to stablish what the best way to **express** a solution is, so that a codebase can be developed by multiple experts and optimized by the compiler.

In this regard, Python does an amazing job with its concept of *Pythonic Code*.
