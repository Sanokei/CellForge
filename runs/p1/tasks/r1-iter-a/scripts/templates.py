# Hand-transcribed 15x12 glyph templates (my own reading of dev PNG dumps).
# X = ink (+1), . = background (-1), o = don't care (0).
# Columns = cell cols 0..11 (col 0 = render x), rows = glyph top .. top+14.
T = {
"0": ["....oXXXXo..", "...oXXXXXXo.", "...XXX..oXX.", "..oXX....XXo", "..oXo....oXo",
      "..oXo....oXX", "..XXo....oXX", "..XXooXXooXX", "..XXooXXooXX", "..oXo....oXX",
      "..oXo....oXo", "..oXX....XXo", "...XXX..XXX.", "...oXXXXXXo.", "....oXXXXo.."],
"1": ["....ooXXX...", "...oXXXXX...", "...oXooXX...", "......oXX...", "......oXX...",
      "......oXX...", "......oXX...", "......oXX...", "......oXX...", "......oXX...",
      "......oXX...", "......oXX...", "......oXX...", "...oXXXXXXXX", "...oXXXXXXXX"],
"2": ["...oXXXXXo..", "..oXXXXXXXo.", "..oXoo.oXXX.", ".........XXo", ".........XXo",
      ".........XX.", "........oXo.", ".......oXX..", "......oXX...", ".....oXXo...",
      "....oXXo....", "...oXXo.....", "..oXXo......", "..oXXXXXXXXo", "..oXXXXXXXXo"],
"3": ["...oXXXXXo..", "..oXXXXXXXo.", "..oXoo.ooXX.", ".........XXo", ".........XX.",
      ".......ooXX.", "....oXXXXX..", "....oXXXXX..", ".......oXXX.", ".........XXo",
      ".........oXo", ".........XXo", "..ooo..ooXXo", "..XXXXXXXXo.", "...oXXXXXo.."],
"4": [".......oXXo.", ".......XXXo.", "......XXXXo.", ".....oXoXXo.", ".....XX.XXo.",
      "....oXo.XXo.", "....Xo..XXo.", "...oX...XXo.", "..oXo...XXo.", "..XX....XXo.",
      "..XXXXXXXXXX", "..XXXXXXXXXX", "........XXo.", "........XXo.", "........XXo."],
"5": ["...XXXXXXXo.", "...XXXXXXXo.", "...XX.......", "...XX.......", "...XX.......",
      "...XXXXXXo..", "...XXXXXXXo.", "...Xo..oXXX.", ".........XXo", ".........XXo",
      ".........XXo", ".........XXo", "..ooo..oXXX.", "..XXXXXXXXo.", "...oXXXXoo.."],
"6": [".....oXXXXo.", "....XXXXXXX.", "...XXXo..oo.", "...XXo......", "..oXX.......",
      "..oXo.......", "..XXooXXXo..", "..XXoXXXXXX.", "..XXXo..oXXo", "..XXX....oXX",
      "..oXo....oXX", "..oXX....oXX", "...XXo..oXXo", "...oXXXXXXX.", "....oXXXXo.."],
"7": ["..XXXXXXXXXo", "..XXXXXXXXXo", ".........XX.", "........oXo.", "........XXo.",
      "........XX..", ".......oXo..", ".......XXo..", "......oXX...", "......oXo...",
      "......XX....", ".....oXX....", ".....XXo....", "....oXX.....", "....oXo....."],
"8": ["....oXXXXo..", "...XXXXXXXX.", "..oXXo..oXXo", "..oXX....XXo", "..oXX....XXo",
      "...XXo..oXX.", "....oXXXXX..", "...oXXXXXXo.", "..oXXo..oXXo", "..oXo....oXo",
      "..XXo....oXX", "..XXo....oXX", "..oXXo..oXXo", "...XXXXXXXX.", "....oXXXXo.."],
"9": ["....oXXXXo..", "...XXXXXXXo.", "..oXXo.ooXX.", "..XXo....XXo", "..XXo....oXo",
      "..XXo....XXo", "..oXXo..oXXX", "...XXXXXXoXX", "....oXXXooXX", ".........oXo",
      ".........XXo", "........oXX.", "...oo..oXXo.", "...XXXXXXX..", "...oXXXXo..."],
}
for k, v in T.items():
    assert len(v) == 15 and all(len(r) == 12 for r in v), k
