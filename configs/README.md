# Road Network Construction
Road network files (`*.net.xml`) are generated using the `netgenerate` software as part of the SUMO software suite. This file details the parameters used for constructing each of the network files.

A key emphasis of this work is to study how *deep reinforcement learning* can be used to provide smart traffic light control when intersections are heterogeneous. Most works assume that the intersections are identical and have the same traffic light phase logic. This is a very simplifying and unrealistic assumption. This work aims to remove this assumption. This detail serves as the foundation of the parameters/arguments chosen for road network generation.

## Assumptions Held Constant
* `--no-turnarounds`: No turnarounds (U-turns). By default, `netgenerate` will produce road networks where U-turns are allowed at all intersections. This greatly complicates congestion and makes catastrophic congestion more likely. On top of that, U-turns are not entirely common and `randomTrips.py` produces route files with far too many U-turns. As such, we remove them as legal flows entirely.
* `-j=traffic_light`: Intersections must be equipped with traffic lights.

***

## Road Networks

### 3x3 Grid Network
> `netgenerate --grid --grid.number=3 --grid.length=150 --no-turnarounds -j=traffic_light -L=2 --output=grid-3x3.net.xml`

### 5x5 Grid Network
> `netgenerate --grid --grid.number=5 --grid.length=150 --no-turnarounds -j=traffic_light -L=2 --output=grid-5x5.net.xml`



***


# New Road Networks (Post-ICCPS Reviews)
Moving forward, we will use the following road networks:

## open grid network
> `netgenerate --grid --grid.number=5 --grid.length=100 --no-turnarounds -j=traffic_light --no-turnarounds --output=grid.net.xml --grid.attach-length=100 -L=3`

## open spider network
> `netgenerate --spider --spider.arm-number=4 --spider.circle-number=3 --spider.omit-center -j=traffic_light -L=3 --spider.space-radius=150 --no-turnarounds --output=spider.net.xml`

## open random network
> `netgenerate --rand --rand.iterations=20 --rand.random-lanenumber=true -L=2 -j=traffic_light --no-turnarounds --output=rand.net.xml`

In addition, we will use real-world traffic evaluation metrics, provided in SUMO, to compare the results. Finally, we will use the baseline timed phase traffic light program that is provided by default using the `netgenerate` command.