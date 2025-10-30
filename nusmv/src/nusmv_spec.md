MODULE main
    IVAR
        y0 : integer;
        z0 : integer;
    VAR
        x : integer;
        y : integer;
        z : integer;
    ASSIGN
        init(x) := 0;
        init(y) := y0;
        init(z) := z0;
        next(x) := (x < y | x < z) ? x + 1 : 0;
        next(y) := y;
        next(z) := z;