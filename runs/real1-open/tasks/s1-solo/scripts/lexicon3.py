# Bulk vocabulary typed by me from general English knowledge of children's stories.
# Regular inflections are expanded by small morphology rules I wrote (not learned from data).
def _double(v):
    return len(v) >= 3 and v[-1] not in "aeiouwxyh" and v[-2] in "aeiou" and v[-3] not in "aeiou" and len(v) <= 4

def past(v):
    if v.endswith('e'): return v + 'd'
    if v.endswith('y') and v[-2] not in 'aeiou': return v[:-1] + 'ied'
    if _double(v): return v + v[-1] + 'ed'
    return v + 'ed'

def ing(v):
    if v.endswith('ie'): return v[:-2] + 'ying'
    if v.endswith('e') and not v.endswith('ee'): return v[:-1] + 'ing'
    if _double(v): return v + v[-1] + 'ing'
    return v + 'ing'

def third(v):
    if v.endswith(('s', 'sh', 'ch', 'x', 'z')): return v + 'es'
    if v.endswith('y') and v[-2] not in 'aeiou': return v[:-1] + 'ies'
    return v + 's'

def plural(n):
    if n.endswith(('s', 'sh', 'ch', 'x', 'z')): return n + 'es'
    if n.endswith('y') and n[-2] not in 'aeiou': return n[:-1] + 'ies'
    if n.endswith('f'): return n[:-1] + 'ves'
    if n.endswith('fe'): return n[:-2] + 'ves'
    return n + 's'

# base verbs by tier weight (regular verbs get -s/-ed/-ing forms)
VERBS = {
30: "play look help want love like walk jump laugh smile call ask start need try watch open turn wish hope",
12: "climb dance clap cook clean paint talk listen learn share hug kiss visit follow chase wave shout cry live stay wait pull push pick miss enjoy decide plan move fix build wash show save bake plant wonder imagine pretend explore discover collect sail shine glow race",
5: "hop skip bounce splash spin roll kick throw toss catch tickle giggle snuggle cuddle whisper yell scream sigh frown grin stare peek point wiggle wobble tumble crawl creep sneak hurry rush float drift melt freeze burn cook stir mix pour spill fill shake rattle knock ring bark meow roar buzz chirp sing hum tap clap pat rub scratch brush comb dress pack carry lift drag stack sort count add fold tie zip cover hide seek search notice remember forget believe trust promise agree refuse allow invite welcome thank greet meet join gather scatter spread reach touch feel taste smell sniff chew lick bite swallow breathe cough sneeze yawn rest relax nap dream wander travel visit arrive return escape rescue protect guard defend attack fight scare frighten surprise amaze delight please bother annoy tease trick fool cheat borrow lend order shoot destroy break crash smash rip tear cut slice chop peel squeeze crush",
2: "admire adore advise appear attend bounce brag bump chat cheer chew chase clip cross damage decorate deliver design disappear drop dust empty enter examine exercise fetch flap flip fry gaze grab guess handle hatch heal hook hunt insist interrupt jog juggle kneel label land lead lock manage marry measure mend mention mumble nod obey offer own paddle park pause peel pinch pity plead poke polish practice praise prepare press pretend print punch race raise rake rely remind repair replace reply scare scoop scrub seem serve settle sew shrink sink slide slip smell snap soak sparkle sprinkle stamp steal stick stretch strike suggest supply swing tackle tempt tiptoe trace trap treat trim twist wag wander warn wrap",
}
# irregular verb forms typed explicitly: (base, 3sg, past, ing, weight)
IRREGULAR = [
 ("go","goes","went","going",60),("see","sees","saw","seeing",50),("come","comes","came","coming",40),
 ("make","makes","made","making",45),("take","takes","took","taking",35),("get","gets","got","getting",40),
 ("give","gives","gave","giving",25),("say","says","said","saying",30),("tell","tells","told","telling",25),
 ("find","finds","found","finding",40),("know","knows","knew","knowing",25),("think","thinks","thought","thinking",25),
 ("feel","feels","felt","feeling",35),("eat","eats","ate","eating",20),("run","runs","ran","running",25),
 ("fly","flies","flew","flying",20),("swim","swims","swam","swimming",10),("sing","sings","sang","singing",15),
 ("sit","sits","sat","sitting",15),("stand","stands","stood","standing",10),("fall","falls","fell","falling",20),
 ("sleep","sleeps","slept","sleeping",12),("wake","wakes","woke","waking",8),("begin","begins","began","beginning",12),
 ("keep","keeps","kept","keeping",10),("leave","leaves","left","leaving",12),("lose","loses","lost","losing",10),
 ("meet","meets","met","meeting",12),("hear","hears","heard","hearing",15),("bring","brings","brought","bringing",12),
 ("buy","buys","bought","buying",8),("catch","catches","caught","catching",10),("throw","throws","threw","throwing",8),
 ("draw","draws","drew","drawing",8),("grow","grows","grew","growing",10),("break","breaks","broke","breaking",8),
 ("ride","rides","rode","riding",8),("speak","speaks","spoke","speaking",5),("wear","wears","wore","wearing",6),
 ("win","wins","won","winning",8),("build","builds","built","building",8),("send","sends","sent","sending",5),
 ("spend","spends","spent","spending",4),("teach","teaches","taught","teaching",5),("hold","holds","held","holding",10),
 ("put","puts","put","putting",20),("let","lets","let","letting",20),("read","reads","read","reading",10),
 ("shake","shakes","shook","shaking",4),("blow","blows","blew","blowing",8),("hide","hides","hid","hiding",8),
 ("sell","sells","sold","selling",3),("pay","pays","paid","paying",3),("lead","leads","led","leading",4),
 ("choose","chooses","chose","choosing",5),("forget","forgets","forgot","forgetting",6),("hurt","hurts","hurt","hurting",6),
 ("cut","cuts","cut","cutting",4),("hit","hits","hit","hitting",4),("dig","digs","dug","digging",5),
 ("hang","hangs","hung","hanging",5),("swing","swings","swung","swinging",3),("shine","shines","shone","shining",3),
 ("wind","winds","wound","winding",1),("bite","bites","bit","biting",3),("feed","feeds","fed","feeding",4),
 ("light","lights","lit","lighting",3),("stick","sticks","stuck","sticking",3),("steal","steals","stole","stealing",3),
 ("do","does","did","doing",60),("have","has","had","having",100),("be","is","was","being",100),
]
# nouns by tier weight (plural forms added with the plural rule)
NOUNS = {
30: "girl boy dog cat bird tree flower friend toy ball house home park day night time way thing",
12: "mom dad man woman baby king queen prince princess fish bear rabbit bunny frog duck mouse pig cow horse lion monkey squirrel fox owl turtle snake bee butterfly bug insect ant worm spider kitten puppy pony cake cookie apple candy pie cup box bag hat shoe dress coat book picture game song story party gift present surprise treasure map key door window room bed table chair floor wall garden forest river lake sea beach sand rock stone hill mountain cave field farm school class store shop town city village street road bridge castle tower boat ship car truck bus train plane bike kite balloon doll teddy drum piano cloud star moon sun sky rain snow wind storm rainbow fire light water grass leaf branch stick seed egg nest wing tail paw fur head hand foot eye ear nose mouth face arm leg finger heart smile tear voice name word lesson idea plan dream wish adventure trip journey race fair zoo circus picnic",
5: "teacher doctor nurse farmer baker cook police pirate wizard witch fairy giant dragon monster ghost robot alien angel elf goblin troll unicorn mermaid dinosaur clown captain sailor soldier knight artist singer dancer driver neighbor stranger visitor guest kid child children people family brother sister grandma grandpa uncle aunt cousin twin pet animal insect kitten cub lamb goat sheep deer wolf tiger elephant giraffe zebra hippo kangaroo panda koala camel penguin parrot eagle hawk crow raven sparrow robin peacock swan goose chicken hen rooster turkey lizard crab whale dolphin shark octopus jellyfish snail beetle ladybug caterpillar grasshopper cricket firefly bat skunk raccoon beaver otter squirrel hamster hedgehog porcupine moose bull donkey pumpkin banana carrot corn pizza soup bread milk juice tea cheese honey sandwich cereal cheese lemon orange grape peach pear cherry berry strawberry watermelon melon potato tomato pea bean nut chocolate sugar salt butter jam lunch dinner breakfast snack meal fruit vegetable candy ice cream sofa couch lamp clock mirror blanket pillow shelf stairs kitchen bedroom bathroom yard fence gate roof attic basement hall paper pencil crayon paint brush scissors glue tape rope string ribbon button zipper pocket wallet coin money penny gold silver diamond jewel ring necklace crown wand sword shield helmet flag ladder wheel engine motor rocket spaceship planet earth world island ocean wave shell pond stream waterfall puddle mud dirt dust flower petal bush vine root trunk log bark acorn pinecone thorn weed hay barn tractor fence pasture meadow valley desert jungle swamp path trail tunnel hole pit well fountain statue museum library hospital church market restaurant bakery playground pool slide swing seesaw sandbox tent campfire backpack lantern flashlight camera phone computer television radio letter card envelope package parcel box basket bucket bottle jar bowl plate spoon fork knife glass mug pot pan oven stove fridge sink bathtub towel soap comb brush toothbrush umbrella boots gloves scarf sweater shirt pants socks pajamas costume mask cape",
2: "acorn anchor angel ankle arrow balloon banner barrel bench blossom bolt bone bottle bubble bucket cabin candle canoe cart cloak clover coach cocoon cone crumb dew dish dot drawer drop dune echo edge engine feather fern flame flute frost gem giggle glitter grain hammer handle harbor hive hoop hut icicle igloo jar jelly jigsaw jug ladder lawn lily lump mat medal melody mitten mist moss nail needle net noodle oar orchard paddle parade pebble pillowcase pirate plank plum pond poster pudding pumpkin puppet puzzle quilt raft rattle riddle rooster saddle sail scarecrow shadow shed shovel siren skate sled slipper soap spark spoon sprout squeak stage stamp steam stew sticker stool straw sunflower tadpole teapot thunder tire toast tomato tongue tooth top trophy tub tulip twig vase vest wagon wig windmill wire yarn",
}
ADJS = {
20: "happy sad big little small old new good bad great nice pretty beautiful bright shiny kind brave scared excited proud sorry surprised sunny warm cold soft hard tall short long fast slow loud quiet high low deep hot wet dry dark light clean dirty empty full heavy funny silly best better special strong young lucky tired hungry angry",
6: "gentle curious careful friendly lonely sleepy sick afraid nervous worried glad shy mean rude lazy clever wise fluffy furry hairy cozy fuzzy scary spooky strange weird tiny huge giant enormous round square flat sharp colorful poor rich wealthy busy bored upset mad grumpy jealous greedy selfish naughty evil wicked polite helpful honest fair delicious yummy tasty sweet sour salty juicy fresh crunchy sticky messy bumpy smooth rough perfect wonderful amazing exciting lovely important favorite whole real ready sure right wrong true alive awake asleep able",
2: "ancient anxious awful brilliant calm cheerful chilly clumsy cloudy colorful comfortable confused cool crooked cruel curly damp dizzy dusty eager elegant fancy fierce foolish fragile freezing frightened gloomy graceful grand grateful grumpy guilty handsome harsh hollow icy jolly lively lonesome loyal magical mighty mysterious narrow noisy odd peaceful pleasant plump polite precious proud puzzled rainy rusty scary scruffy selfish shaggy shiny shy silent sleepy slimy slippery snowy sparkly spotted squeaky steady stormy stubborn sunny tender terrible thick thin thirsty thoughtful tidy tough tricky ugly unhappy unusual wild windy wobbly worried",
}
ADVS = {
6: "quickly slowly carefully gently loudly quietly softly happily sadly proudly bravely suddenly finally soon later early late again always never sometimes often usually already almost even ever far near close hard fast well",
2: "angrily anxiously badly bitterly boldly brightly calmly cheerfully clearly closely cruelly curiously eagerly easily excitedly fiercely gladly gracefully greedily hungrily kindly lazily lightly madly nervously noisily patiently politely rapidly sharply shyly silently sweetly tightly warmly wildly",
}

def build():
    w = {}
    def put(k, wt):
        w[k] = max(w.get(k, 0), float(wt))
    for wt, s in VERBS.items():
        for v in s.split():
            put(v, wt*0.55); put(third(v), wt*0.25); put(past(v), wt*0.9); put(ing(v), wt*0.5)
    for b, t3, p, g, wt in IRREGULAR:
        put(b, wt*0.55); put(t3, wt*0.25); put(p, wt*0.9); put(g, wt*0.5)
    for wt, s in NOUNS.items():
        for n in s.split():
            put(n, wt); put(plural(n), wt*0.45)
    for wt, s in ADJS.items():
        for a in s.split(): put(a, wt)
    for wt, s in ADVS.items():
        for a in s.split(): put(a, wt)
    return w
