Module {
    extl: [
        Wire {
            vec: [
                (
                    3,
                    Int,
                ),
                (
                    4,
                    Int,
                ),
            ],
        },
        Wire {
            vec: [
                (
                    8,
                    Int,
                ),
                (
                    9,
                    Int,
                ),
            ],
        },
    ],
    intf: [
        Wire {
            vec: [
                (
                    0,
                    Int,
                ),
                (
                    1,
                    Int,
                ),
                (
                    2,
                    Int,
                ),
            ],
        },
        Wire {
            vec: [
                (
                    5,
                    Int,
                ),
                (
                    6,
                    Int,
                ),
                (
                    7,
                    Int,
                ),
            ],
        },
    ],
    prvt: [
        Wire {
            vec: [],
        },
        Wire {
            vec: [],
        },
    ],
    ctrl: [
        Wire {
            vec: [
                (
                    0,
                    Int,
                ),
                (
                    1,
                    Int,
                ),
                (
                    2,
                    Int,
                ),
            ],
        },
        Wire {
            vec: [
                (
                    5,
                    Int,
                ),
                (
                    6,
                    Int,
                ),
                (
                    7,
                    Int,
                ),
            ],
        },
    ],
    obs: [
        Wire {
            vec: [
                (
                    0,
                    Int,
                ),
                (
                    1,
                    Int,
                ),
                (
                    2,
                    Int,
                ),
                (
                    3,
                    Int,
                ),
                (
                    4,
                    Int,
                ),
            ],
        },
        Wire {
            vec: [
                (
                    5,
                    Int,
                ),
                (
                    6,
                    Int,
                ),
                (
                    7,
                    Int,
                ),
                (
                    8,
                    Int,
                ),
                (
                    9,
                    Int,
                ),
            ],
        },
    ],
    wire: [
        Wire {
            vec: [
                (
                    0,
                    Int,
                ),
                (
                    1,
                    Int,
                ),
                (
                    2,
                    Int,
                ),
                (
                    3,
                    Int,
                ),
                (
                    4,
                    Int,
                ),
            ],
        },
        Wire {
            vec: [
                (
                    5,
                    Int,
                ),
                (
                    6,
                    Int,
                ),
                (
                    7,
                    Int,
                ),
                (
                    8,
                    Int,
                ),
                (
                    9,
                    Int,
                ),
            ],
        },
    ],
    atoms: [
        Atom {
            ctrl: Wire {
                vec: [
                    (
                        5,
                        Int,
                    ),
                    (
                        6,
                        Int,
                    ),
                    (
                        7,
                        Int,
                    ),
                ],
            },
            wait: Wire {
                vec: [
                    (
                        8,
                        Int,
                    ),
                    (
                        9,
                        Int,
                    ),
                ],
            },
            read: Wire {
                vec: [
                    (
                        0,
                        Int,
                    ),
                    (
                        1,
                        Int,
                    ),
                    (
                        2,
                        Int,
                    ),
                ],
            },
            init: [
                Term {
                    itype: ConstInt(
                        0,
                    ),
                    write: Wire {
                        vec: [
                            (
                                5,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [],
                    },
                },
                Term {
                    itype: Abs,
                    write: Wire {
                        vec: [
                            (
                                16,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                8,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Assign,
                    write: Wire {
                        vec: [
                            (
                                6,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                16,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Abs,
                    write: Wire {
                        vec: [
                            (
                                17,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                9,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Assign,
                    write: Wire {
                        vec: [
                            (
                                7,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                17,
                                Int,
                            ),
                        ],
                    },
                },
            ],
            update: [
                Term {
                    itype: Lt,
                    write: Wire {
                        vec: [
                            (
                                10,
                                Bool,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                0,
                                Int,
                            ),
                            (
                                1,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Lt,
                    write: Wire {
                        vec: [
                            (
                                11,
                                Bool,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                0,
                                Int,
                            ),
                            (
                                2,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Or,
                    write: Wire {
                        vec: [
                            (
                                12,
                                Bool,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                10,
                                Bool,
                            ),
                            (
                                11,
                                Bool,
                            ),
                        ],
                    },
                },
                Term {
                    itype: ConstInt(
                        1,
                    ),
                    write: Wire {
                        vec: [
                            (
                                13,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [],
                    },
                },
                Term {
                    itype: Add,
                    write: Wire {
                        vec: [
                            (
                                14,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                0,
                                Int,
                            ),
                            (
                                13,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: ConstInt(
                        0,
                    ),
                    write: Wire {
                        vec: [
                            (
                                15,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [],
                    },
                },
                Term {
                    itype: Cond,
                    write: Wire {
                        vec: [
                            (
                                5,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                12,
                                Bool,
                            ),
                            (
                                14,
                                Int,
                            ),
                            (
                                15,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Assign,
                    write: Wire {
                        vec: [
                            (
                                6,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                1,
                                Int,
                            ),
                        ],
                    },
                },
                Term {
                    itype: Assign,
                    write: Wire {
                        vec: [
                            (
                                7,
                                Int,
                            ),
                        ],
                    },
                    read: Wire {
                        vec: [
                            (
                                2,
                                Int,
                            ),
                        ],
                    },
                },
            ],
        },
    ],
}