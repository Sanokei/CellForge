# Hand-typed word classes and class-transition weights (my own English knowledge).
GROUPS = ['DET','PRON','AUX','PREP','CONJ','ADV','ADJ','NAME','VPAST','VBASE','VING','NOUN','TO','QT']
G = len(GROUPS)
GI = {g: i for i, g in enumerate(GROUPS)}
# class-state dims: the 13 word groups (QT excluded) + sentence-start, after-comma, after-open-quote
STATES = GROUPS[:13] + ['S', 'CM', 'Q']
K = len(STATES)
SI = {s: i for i, s in enumerate(STATES)}

DET = set("the a an his her their my your our its this that these those some many every each all no another any both several".split())
PRON = set("he she they it i we you him them me us who what which".split())
AUX = set("was were is are am had has have did do does can could would will should might must be been being".split())
PREP = set("in on at with of for from by about into up down over out around through under near after before like as than off across behind beside between during without".split())
CONJ = set("and but so then when because if while or until since though although".split())
ADV = set("very too really just also still always never soon suddenly finally now often maybe again together away here there back even only much more most once twice sometimes yesterday today tomorrow everywhere ago already almost enough anymore yes no oh wow hello hi please thank goodbye bye okay yay hooray oops ouch hey sorry thanks welcome not".split())
FAMILY = set("mom dad mommy daddy grandma grandpa mama papa".split())
VBASE = set("""play go see make help get find take want love like come look give tell say know think feel try need eat run walk jump fly swim climb sing dance draw paint read write learn show share hide catch throw keep let put bring open close stop start turn ask call hear listen stay live wait watch meet visit clean wash cook build fix fall sleep wake buy sell pick pull push hold hug kiss laugh cry smile shout remember pretend imagine wish hope believe follow plant grow win lose choose sit stand move change enjoy explore discover talk work
plays goes sees makes helps gets finds takes wants loves likes comes looks gives tells says knows thinks feels tries needs eats runs walks jumps flies swims climbs sings dances draws paints reads writes learns shows shares hides catches throws keeps lets puts brings opens closes stops starts turns asks calls hears listens stays lives waits watches meets visits cleans washes cooks builds fixes falls sleeps wakes""".split())
IRREG = set("went saw came took gave made felt said found got knew thought told ran sat stood fell flew swam sang ate drank slept woke began held kept left lost met heard brought bought caught threw drew grew broke rode spoke wore won built sent spent taught read put hit cut shook became sold hid stuck swung lit blew fed led paid dug".split())
NOT_ED = set("red bed need feed seed speed weed hundred sacred naked shed sled bread head thread spread ahead instead".split())
NOT_ING = set("morning something everything nothing anything king ring spring sing thing bring swing wing string evening darling ceiling building".split())
ADJ_EXTRA = set("""happy sad big little small old new good bad great nice pretty beautiful bright shiny red blue green yellow pink purple black white orange brown kind brave scared excited proud sorry surprised sunny warm cold soft hard tall short long fast slow loud quiet high low deep hot wet dry dark light clean dirty empty full heavy funny silly best better special strong young lucky tired hungry angry gentle curious careful friendly lonely sleepy sick hurt afraid nervous worried glad shy mean rude lazy clever wise fluffy furry hairy cozy fuzzy scary spooky strange weird tiny huge giant enormous round square flat sharp colorful poor rich wealthy busy bored upset mad grumpy jealous greedy selfish naughty evil wicked polite helpful honest fair delicious yummy tasty sweet sour salty juicy fresh crunchy sticky messy bumpy smooth rough perfect wonderful amazing exciting lovely important favorite whole real ready sure right wrong true alive awake asleep able own same different other next last first only second third""".split())
NOUN_FORCE = set("day time night morning story family party".split())


def classify(w, names):
    if w == '"': return 'QT'
    if w in names or w in FAMILY: return 'NAME'
    if w == 'to': return 'TO'
    if w in DET: return 'DET'
    if w in PRON: return 'PRON'
    if w in AUX: return 'AUX'
    if w in CONJ: return 'CONJ'
    if w in PREP: return 'PREP'
    if w in ADV: return 'ADV'
    if w in NOUN_FORCE: return 'NOUN'
    if w in VBASE: return 'VBASE'
    if w in IRREG: return 'VPAST'
    if w.endswith('ing') and w not in NOT_ING and len(w) > 4: return 'VING'
    if w.endswith('ed') and w not in NOT_ED and len(w) > 3: return 'VPAST'
    if w.endswith('ly') and len(w) > 4: return 'ADV'
    if w in ADJ_EXTRA: return 'ADJ'
    return 'NOUN'

# next-group weights given the previous class (rows normalised by the generator).
# order: DET PRON AUX PREP CONJ ADV ADJ NAME VPAST VBASE VING NOUN TO QT
PI = {
 'DET':  [1,0.5,0,0,0,2,38,1,1,0,1,55,0,0],
 'PRON': [3,2,18,10,7,5,2,0,40,10,0,2,3,0],
 'AUX':  [17,2,3,8,1,12,30,2,6,5,12,6,3,0],
 'PREP': [42,10,0,1,0,1,8,8,0,2,5,13,0,0],
 'CONJ': [20,38,3,2,0,6,4,15,6,0,0,5,0,0],
 'ADV':  [5,4,2,10,2,8,30,0,20,4,5,5,2,0],
 'ADJ':  [1,1,2,6,16,2,8,0,3,0,0,48,0,0],
 'NAME': [1,1,17,6,13,4,0,0,40,2,1,2,0,0],
 'VPAST':[27,7,0,25,10,8,4,3,0,2,4,5,12,0],
 'VBASE':[27,8,0,20,7,8,4,4,0,0,2,8,2,0],
 'VING': [18,5,0,30,8,10,5,3,0,0,0,8,3,0],
 'NOUN': [2,2,17,18,17,3,0,2,22,2,0,4,0,0],
 'TO':   [17,3,0,0,0,2,0,5,0,70,0,3,0,0],
 'S':    [28,34,0,4,8,5,1,18,0,0,0,1,0,3],
 'CM':   [15,26,0,6,22,6,2,10,2,0,0,2,0,5],
 'Q':    [8,35,5,1,3,10,1,12,0,15,0,4,0,0],
}
