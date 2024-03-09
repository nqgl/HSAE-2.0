So I'm currently like "ah shit, I overengineered it maybe"

let's see, well so here's the alternate design

ActTracker(nn.module)
Encoder
Decoder

SAE ~= nn.Sequential(
    [
        Encoder(),
        ActTracker(),
        Decoder(),
    ]
) * not actually quite this simple though
    - tracker needs to have a ref to the weights it is talking about
    - parent needs to have ref to all its trackers
- requires more epicycles or requires storing enc.cached_xyz values
    - and I kind of hate that, it seems pretty incorrect (this is why I made the Cache)





Cases I'm trying to handle:

- Grab l1, l0 loss from forward pass elegantly
- Different types of forward pass: (X denoting n-tensor where n>2)
    - forward(x)
    - forward(x, gate)
    - forward(X)
    - forward(X, gate)
- Different resampling methods
    - like 4+
    - ghost grads might be very weird. 
        - maybe I should maximally work towards making ghost grads make sense bc it's the one I have thought about the least
- Different activation calculation methods
    - currently, 2
- Different architectural cases
    - I don't want this to be wedded to SAEs. At a minimum I intend for the framework to support:
        - Hierarchical/MOE saes
        - Multi-layer networks



Addressing these in the ComponentLayer strategy:

- Grab l1, l0 loss from forward pass elegantly
+ this is currently happening pretty elegantly, I think
+ but not necessarily max efficiency. For one I prob can't .compile this :/
+ on the other hand, I don't really care about caching stuff other than acts, so the whole core of the FP could be it's own compiled module


- Different types of forward pass: (X denoting n-tensor where n>2)
    - forward(x)
    - forward(x, gate)
    - forward(X)
    - forward(X, gate)
    + n>2 - tensor support seems solid in CL strat.\
    + that's not necessarily the main part of the problem though.
    + what would it look like with a gated fwd in there? not totally sure if that goess smooth or is just as much a pain
    + was thinking about gate ending up in cache.prev_cache.gate or something
        or just cache.gate
        using either write or read hooks
- Different resampling methods
    - like 4+
+   - ghost grads might be very weird. +++
        - maybe I should maximally work towards making ghost grads make sense bc it's the one I have thought about the least
    - what about the thing I was thinking about doing. how would that go?
        - CacheLayer nonlinearity could come from the resetting module or something
- Different activation calculation methods
    - currently, 2
    + this is easy with the current approach
- Different architectural cases
    - I don't want this to be wedded to SAEs. At a minimum I intend for the framework to support:
        - Hierarchical/MOE saes
        - Multi-layer networks
    + multi layer seems 






Addressing these in a more toned down strategy:

- Grab l1, l0 loss from forward pass elegantly
    + sounds like most ways to do this are inelegant (one design problem ive been grappling with since the beginning)
    + elegant here might would be like "we have a new returntype"
    + and maybe a custom nn.sequential that supports it
    + wherein you put returns like in named fields or something?? and then have rules about what gets put into the next module from this one and what are like layer outputs to store for later? hmm idk about that
        --- yeah that's *gross*
    + oh well it could again just be like a sequential that sticks a cache into every fp (or at least every fp that accepts one)
        - mostly not so gross
    + is that not what we already have..? maybe original approach is okay?
    + then you have to do something with those cached acts... 
    + so like, you have an actprocessor, and it executes operations on the acts, and some of those end up in the cache, and some end up getting calculated and being inside the module that like calculates freqs.
        + though, it could do this with cache hooks instead of by being like a component
        + uh and how do you propose to register those hooks with every cache? the bottom level layer should not care or be responsible for that, and should just take a cache and write to it
        + then a higher level layer should insert a cache with the right hooks...
        +++ literally what it does right now
        + though maybe it could be an even higher level layer?  
- Different types of forward pass: (X denoting n-tensor where n>2)
    - forward(x)
    - forward(x, gate)
    - forward(X)
    - forward(X, gate)
- Different resampling methods
    - like 4+
    - ghost grads might be very weird. 
        - maybe I should maximally work towards making ghost grads make sense bc it's the one I have thought about the least
- Different activation calculation methods
    - currently, 2
- Different architectural cases
    - I don't want this to be wedded to SAEs. At a minimum I intend for the framework to support:
        - Hierarchical/MOE saes
        - Multi-layer networks





