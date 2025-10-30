Module {
    extl: [
        Wire {
            ranges: [
                [3, 4] : Int,
            ],
        },
        Wire {
            ranges: [
                [8, 9] : Int,
            ],
        },
    ],
    intf: [
        Wire {
            ranges: [
                [0, 2] : Int,
            ],
        },
        Wire {
            ranges: [
                [5, 7] : Int,
            ],
        },
    ],
    prvt: [
        Wire {
            ranges: [],
        },
        Wire {
            ranges: [],
        },
    ],
    ctrl: [
        Wire {
            ranges: [
                [0, 2] : Int,
            ],
        },
        Wire {
            ranges: [
                [5, 7] : Int,
            ],
        },
    ],
    obs: [
        Wire {
            ranges: [
                [0, 4] : Int,
            ],
        },
        Wire {
            ranges: [
                [5, 9] : Int,
            ],
        },
    ],
    wire: [
        Wire {
            ranges: [
                [0, 4] : Int,
            ],
        },
        Wire {
            ranges: [
                [5, 9] : Int,
            ],
        },
    ],
    atoms: [
        Atom {
            ctrl: Wire {
                ranges: [
                    [5, 7] : Int,
                ],
            },
            wait: Wire {
                ranges: [
                    [8, 9] : Int,
                ],
            },
            read: Wire {
                ranges: [
                    [0, 2] : Int,
                ],
            },
            init: [
                Term {
                    itype: Assign(
                        VarRef(
                            "x",
                        ),
                        ConstInt(
                            0,
                        ),
                    ),
                    write: Wire {
                        ranges: [
                            [5, 5] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [],
                    },
                },
                Term {
                    itype: Assign(
                        VarRef(
                            "y",
                        ),
                        VarRef(
                            "y0",
                        ),
                    ),
                    write: Wire {
                        ranges: [
                            [6, 6] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [3, 3] : Int,
                        ],
                    },
                },
                Term {
                    itype: Assign(
                        VarRef(
                            "z",
                        ),
                        VarRef(
                            "z0",
                        ),
                    ),
                    write: Wire {
                        ranges: [
                            [7, 7] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [4, 4] : Int,
                        ],
                    },
                },
            ],
            update: [
                Term {
                    itype: Cond(
                        Or(
                            Lt(
                                VarRef(
                                    "x",
                                ),
                                VarRef(
                                    "y",
                                ),
                            ),
                            Lt(
                                VarRef(
                                    "x",
                                ),
                                VarRef(
                                    "z",
                                ),
                            ),
                        ),
                        Add(
                            VarRef(
                                "x",
                            ),
                            ConstInt(
                                1,
                            ),
                        ),
                        ConstInt(
                            0,
                        ),
                    ),
                    write: Wire {
                        ranges: [
                            [5, 5] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [0, 2] : Int,
                        ],
                    },
                },
                Term {
                    itype: VarRef(
                        "y",
                    ),
                    write: Wire {
                        ranges: [
                            [6, 6] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [1, 1] : Int,
                        ],
                    },
                },
                Term {
                    itype: VarRef(
                        "z",
                    ),
                    write: Wire {
                        ranges: [
                            [7, 7] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [2, 2] : Int,
                        ],
                    },
                },
            ],
        },
    ],
}