# Road Network Construction
Road network files (`*.net.xml`) are generated using the `netgenerate` software as part of the SUMO software suite. This file details the parameters used for constructing each of the network files.

A key emphasis of this work is to study how *deep reinforcement learning* can be used to provide smart traffic light control when intersections are heterogeneous. Most works assume that the intersections are identical and have the same traffic light phase logic. This is a very simplifying and unrealistic assumption. This work aims to remove this assumption. This detail serves as the foundation of the parameters/arguments chosen for road network generation.

## Assumptions Held Constant
* `--no-turnarounds`: No turnarounds (U-turns). By default, `netgenerate` will produce road networks where U-turns are allowed at all intersections. This greatly complicates congestion and makes catastrophic congestion more likely. On top of that, U-turns are not entirely common and `randomTrips.py` produces route files with far too many U-turns. As such, we remove them as legal flows entirely.
* `-j=traffic_light`: Intersections must be equipped with traffic lights.

## Road Networks

### 2x2 Grid Network
> `netgenerate --grid --grid.number=2 --grid.length=100 --no-turnarounds -j=traffic_light -L=3 --output=grid.net.xml`

### 3x3 Grid Network
> `netgenerate --grid --grid.number=3 --grid.length=100 --no-turnarounds -j=traffic_light --rand.random-lanenumber=true -L=4 --output=grid.net.xml`

### 5x5 Grid Network
> `netgenerate --grid --grid.number=5 --grid.length=100 --no-turnarounds -j=traffic_light --rand.random-lanenumber=true -L=4 --output=grid.net.xml`

### 7x7 Grid Network
> `netgenerate --grid --grid.number=7 --grid.length=100 --no-turnarounds -j=traffic_light --rand.random-lanenumber=true -L=4 --output=grid.net.xml`

### 9x9 Grid Network
> `netgenerate --grid --grid.number=9 --grid.length=100 --no-turnarounds -j=traffic_light --rand.random-lanenumber=true -L=4 --output=grid.net.xml`