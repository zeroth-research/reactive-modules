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
                    itype: ConstInt(
                        0,
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
                    itype: Assign,
                    write: Wire {
                        ranges: [
                            [6, 6] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [8, 8] : Int,
                        ],
                    },
                },
                Term {
                    itype: Assign,
                    write: Wire {
                        ranges: [
                            [7, 7] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [9, 9] : Int,
                        ],
                    },
                },
            ],
            update: [
                Term {
                    itype: Lt,
                    write: Wire {
                        ranges: [
                            [10, 10] : Bool,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [0, 1] : Int,
                        ],
                    },
                },
                Term {
                    itype: Lt,
                    write: Wire {
                        ranges: [
                            [11, 11] : Bool,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [0, 0] : Int,
                            [2, 2] : Int,
                        ],
                    },
                },
                Term {
                    itype: Or,
                    write: Wire {
                        ranges: [
                            [12, 12] : Bool,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [10, 11] : Bool,
                        ],
                    },
                },
                Term {
                    itype: ConstInt(
                        0,
                    ),
                    write: Wire {
                        ranges: [
                            [15, 15] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [],
                    },
                },
                Term {
                    itype: ConstInt(
                        1,
                    ),
                    write: Wire {
                        ranges: [
                            [13, 13] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [],
                    },
                },
                Term {
                    itype: Add,
                    write: Wire {
                        ranges: [
                            [14, 14] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [0, 0] : Int,
                            [13, 13] : Int,
                        ],
                    },
                },
                Term {
                    itype: Cond,
                    write: Wire {
                        ranges: [
                            [5, 5] : Int,
                        ],
                    },
                    read: Wire {
                        ranges: [
                            [12, 12] : Bool,
                            [14, 15] : Int,
                        ],
                    },
                },
                Term {
                    itype: Assign,
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
                    itype: Assign,
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