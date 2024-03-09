Schedulable Configs
Cache
CacheLayer
CacheProcLayer
ComponentLayer
    <-CacheProcLayer
LayerComponent



# name binding vs requires/provides component behavior
    
- I don't like the way it works atm where the components bind to some name on the parent layer and can access each other through that interface
- I think instead, they should maybe "require" and "provide" different functionalities/interfaces/endpoints/protocols, and then the parent class can 
- ah, but you still might want these to have some name on the parent class


what about
```python
class SpecificComponentSpec1(ABC/Protocol): ...
class SpecificComponentSpec2(ABC/Protocol): ...

class SCImpl1(SpecificComponentSpec1):
    requires: Dict[Protocol] = {"bindname1":Protocol1, ...}
    provides: Dict[Protocol] = {"bindname1":Protocol1, ...}
class SCImpl2(SpecificComponentSpec2): ...



class Plugins:
    ...
    def external_provide()
class SpecificPluginsSpec(Plugins):
   component1name: SpecificComponentSpec1
   component2name: SpecificComponentSpec2

plugins = SpecificPluginsSpec(SCImpl1(), SCImpl2())
cachelayer = CacheLayer(...)
layer = ComponentLayer(cachelayer, plugins)
```

oh in this framework, also the "_update_from_cache" method could just be a requirement on the components, provided by the ComponentLayer

types of require:
- requring an instance be injected into one of the fields
    - have this as a specific case with "call function x with some specific instance during initialization"
    - and then a decorator that you wrap with that that is like "set this field to the value this function is called with" and also "freak out if you get called twice"
- requiring a callable that implements some behavior
    - think you want this even if you have the instances thing
- requiring a class constructor to make a new component (do we ever want this)
- parent-external name-binding
    @bind_parent_attr.attrname()

- require method (or class?) to be wrapped by another component
- eg this way you could have a CacheProvider protocol and then expose cache requiring functions as

        @injected_wrap(CacheProvider)
- require a named binding    


annotations:
    - call after setup is done
    - call with parent
    




other stuff:
- you can't autowire if a group has the same behavior implementation provided multiple times
    - though you could have duplicate event creators?
- implicit injection could be risky?
- ordering of components for when multiple comp consume the same event





## Cache

A cache is a nice fix to the "how to I get this data out of the forward pass without doing something gross & hacky & kind of wrong. The basic version is nice but has some hazards. I noticed some advanced features + capabilities that would be nice, but which would be VERY hazardous. Hence current design philosophy of the cache:

- Have a bunch of crazy, OP, dangerous capabilities in the cache
    - these are powerful but on their own prone to side effects, spaghetti, mistakes, and hazards
- Aggressively fail if anything weird happens, to help ensure that nothing weird happens and that it's harder to create fucked systems from the get-go
    - You can only ever write once to any non-private field!
        - field rendering must not compete
        - also, don't allow writes to any rendered field 
        - possibly could allow multiple writes, but require that all non-nulls are the same thing or equal


delattr implemented? maybe good, if you're like "this attr should be used once only" then you use it and then del the thing





## New ideas:

component method with @cacherequiring decorator get that argument provided by the container?

re:    parent-external name-binding
        @bind_parent_attr.attrname()
- also want to be able to be like 'this is one of the things that should get called when parent.attrname is called
- implements the thing that @call_on_sublayers did, but does so bottom up instead of top-down
- still a different type of thing to nn.sequential though

->> what to do about nested groups?
    maybe unprovided reqs could be provided by other adjacent groups? seems kinda sketchy though


## resampling separation of responsiblities

<-FT-><-Re-><-RA->
FreqTracker
- job is just to calculate freqs and expose a freqs property
- check if dead, but thresholds for dead are not in this config
- reset freqs for a neuron
Resampler


responsibilities:
- track freq
- hold weights & reset them to new values
- decide which neurons are dead and when 
    + hold a config for this
- implement the specific resampling algorithm
- reset freqs when neuron has been resampled
- 

one idea is like:
Resampler
- hold weights, reset
- send dead to RA

RA:
- get weights, do whatever this RA alg says to do
- then send them back to the Resampler
    - resample and reset freqs
- 







#Other ideas
## Another re-initialization idea

perform SVD on batches, resample to some of the top directions



## Other
only norm the decoder vecs that are too long?
    - not sure this actually helps
if it were a sparse mlp, re-init W 