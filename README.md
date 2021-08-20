# Two-Stage-TSP-in-RC

Entries in [Amazon Last Mile Routing  RESEARCH CHALLENGE](https://routingchallenge.mit.edu/)



## Introduction

We find out that **high-quality routes tend to visit the stops in the same zone before moving onto the next zone**, so we solve the challenge as a special Travel Salesman Problem which we call Two-stage TSP. In the first stage, we sort the zone. In the second stage, we sort the stop in the zone.



We use greedy algorithm, Local Search and Dynamic Programming to solve TSP, so we don't need `model_build`.

