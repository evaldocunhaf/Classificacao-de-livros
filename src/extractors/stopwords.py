STOPWORDS_PT = (
    "a as o os e é de do da dos das em no na nos nas um uma uns umas por para com sem "
    "sob sobre entre depois antes até após desde que quem cujo cuja cujos cujas qual "
    "quais este esta estes estas esse essa esses essas aquele aquela aqueles aquelas "
    "isto isso aquilo eu tu ele ela nós vós eles elas meu minha meus minhas teu tua "
    "teus tuas seu sua seus suas nosso nossa nossos nossas vosso vossa vossos vossas "
    "lhe lhes me te nos vos se si mesmo mesma mesmos mesmas já ainda muito pouco "
    "muitos poucas poucos também não mais menos ao aos à às pelo pela pelos pelas "
    "tem têm hão haja hajam era eram éramos será seriam sendo ser sido estar estava "
    "estávamos tenho temos tinha tínhamos houve houver houveram foi foram fui fomos "
    "seja sejam sejamos onde quando como porque porquê quão quanto quanta quantos "
    "quantas contudo todavia porém entretanto portanto assim apenas somente salvo "
    "exceto acerca dentro fora através durante mediante"
).split()


STOPWORDS_EN = (
    "a an the and or but if then else while for in on at to from by with without of "
    "is are was were be been being have has had do does did will would can could shall "
    "should may might must about above across after against along amid among around as "
    "at before behind below beneath beside between beyond concerning despite down "
    "during except from inside into like near off onto out outside over past regarding "
    "since through throughout till toward under underneath until up upon within this "
    "that these those it its they them their our your my me we you he she his her i "
    "not no nor so very just only own same such than too also once here there when "
    "where why how all any both each few more most other some what which who whom"
).split()

STOPWORDS = set(STOPWORDS_PT) | set(STOPWORDS_EN)