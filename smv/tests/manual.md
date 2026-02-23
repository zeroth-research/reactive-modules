Module {
    extl: Interface {
        wires: [
            [
                Wire {
                    id: 3,
                    dtype: Int,
                },
                Wire {
                    id: 4,
                    dtype: Int,
                },
            ],
            [
                Wire {
                    id: 8,
                    dtype: Int,
                },
                Wire {
                    id: 9,
                    dtype: Int,
                },
            ],
        ],
    },
    intf: Interface {
        wires: [
            [
                Wire {
                    id: 0,
                    dtype: Int,
                },
                Wire {
                    id: 1,
                    dtype: Int,
                },
                Wire {
                    id: 2,
                    dtype: Int,
                },
            ],
            [
                Wire {
                    id: 5,
                    dtype: Int,
                },
                Wire {
                    id: 6,
                    dtype: Int,
                },
                Wire {
                    id: 7,
                    dtype: Int,
                },
            ],
        ],
    },
    prvt: Interface {
        wires: [
            [],
            [],
        ],
    },
    obs: Interface {
        wires: [
            [
                Wire {
                    id: 0,
                    dtype: Int,
                },
                Wire {
                    id: 1,
                    dtype: Int,
                },
                Wire {
                    id: 2,
                    dtype: Int,
                },
                Wire {
                    id: 3,
                    dtype: Int,
                },
                Wire {
                    id: 4,
                    dtype: Int,
                },
            ],
            [
                Wire {
                    id: 5,
                    dtype: Int,
                },
                Wire {
                    id: 6,
                    dtype: Int,
                },
                Wire {
                    id: 7,
                    dtype: Int,
                },
                Wire {
                    id: 8,
                    dtype: Int,
                },
                Wire {
                    id: 9,
                    dtype: Int,
                },
            ],
        ],
    },
    ctrl: Interface {
        wires: [
            [
                Wire {
                    id: 0,
                    dtype: Int,
                },
                Wire {
                    id: 1,
                    dtype: Int,
                },
                Wire {
                    id: 2,
                    dtype: Int,
                },
            ],
            [
                Wire {
                    id: 5,
                    dtype: Int,
                },
                Wire {
                    id: 6,
                    dtype: Int,
                },
                Wire {
                    id: 7,
                    dtype: Int,
                },
            ],
        ],
    },
    temp: Interface {
        wires: [
            [
                Wire {
                    id: 10,
                    dtype: Bool,
                },
                Wire {
                    id: 11,
                    dtype: Bool,
                },
                Wire {
                    id: 12,
                    dtype: Bool,
                },
                Wire {
                    id: 13,
                    dtype: Int,
                },
                Wire {
                    id: 14,
                    dtype: Int,
                },
                Wire {
                    id: 15,
                    dtype: Int,
                },
            ],
        ],
    },
    atoms: [
        Atom {
            ctrl: Interface {
                wires: [
                    [
                        Wire {
                            id: 5,
                            dtype: Int,
                        },
                        Wire {
                            id: 6,
                            dtype: Int,
                        },
                        Wire {
                            id: 7,
                            dtype: Int,
                        },
                    ],
                ],
            },
            wait: Interface {
                wires: [
                    [
                        Wire {
                            id: 8,
                            dtype: Int,
                        },
                        Wire {
                            id: 9,
                            dtype: Int,
                        },
                    ],
                ],
            },
            read: Interface {
                wires: [
                    [
                        Wire {
                            id: 0,
                            dtype: Int,
                        },
                        Wire {
                            id: 1,
                            dtype: Int,
                        },
                        Wire {
                            id: 2,
                            dtype: Int,
                        },
                    ],
                ],
            },
            temp: Interface {
                wires: [
                    [
                        Wire {
                            id: 10,
                            dtype: Bool,
                        },
                        Wire {
                            id: 11,
                            dtype: Bool,
                        },
                        Wire {
                            id: 12,
                            dtype: Bool,
                        },
                        Wire {
                            id: 13,
                            dtype: Int,
                        },
                        Wire {
                            id: 14,
                            dtype: Int,
                        },
                        Wire {
                            id: 15,
                            dtype: Int,
                        },
                    ],
                ],
            },
            init: Block {
                terms: [
                    Term {
                        itype: ConstInt(
                            0,
                        ),
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 5,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [],
                            ],
                        },
                    },
                    Term {
                        itype: Assign,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 6,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 8,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: Assign,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 7,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 9,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                ],
                read: Interface {
                    wires: [
                        [
                            Wire {
                                id: 8,
                                dtype: Int,
                            },
                            Wire {
                                id: 9,
                                dtype: Int,
                            },
                        ],
                    ],
                },
                write: Interface {
                    wires: [
                        [
                            Wire {
                                id: 5,
                                dtype: Int,
                            },
                            Wire {
                                id: 5,
                                dtype: Int,
                            },
                            Wire {
                                id: 6,
                                dtype: Int,
                            },
                            Wire {
                                id: 6,
                                dtype: Int,
                            },
                            Wire {
                                id: 7,
                                dtype: Int,
                            },
                            Wire {
                                id: 7,
                                dtype: Int,
                            },
                        ],
                    ],
                },
            },
            update: Block {
                terms: [
                    Term {
                        itype: Lt,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 10,
                                        dtype: Bool,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 0,
                                        dtype: Int,
                                    },
                                    Wire {
                                        id: 1,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: Lt,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 11,
                                        dtype: Bool,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 0,
                                        dtype: Int,
                                    },
                                    Wire {
                                        id: 2,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: Or,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 12,
                                        dtype: Bool,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 10,
                                        dtype: Bool,
                                    },
                                    Wire {
                                        id: 11,
                                        dtype: Bool,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: ConstInt(
                            1,
                        ),
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 13,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [],
                            ],
                        },
                    },
                    Term {
                        itype: Add,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 14,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 0,
                                        dtype: Int,
                                    },
                                    Wire {
                                        id: 13,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: ConstInt(
                            0,
                        ),
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 15,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [],
                            ],
                        },
                    },
                    Term {
                        itype: Cond,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 5,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 12,
                                        dtype: Bool,
                                    },
                                    Wire {
                                        id: 14,
                                        dtype: Int,
                                    },
                                    Wire {
                                        id: 15,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: Assign,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 6,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 1,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                    Term {
                        itype: Assign,
                        write: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 7,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                        read: Interface {
                            wires: [
                                [
                                    Wire {
                                        id: 2,
                                        dtype: Int,
                                    },
                                ],
                            ],
                        },
                    },
                ],
                read: Interface {
                    wires: [
                        [
                            Wire {
                                id: 0,
                                dtype: Int,
                            },
                            Wire {
                                id: 1,
                                dtype: Int,
                            },
                            Wire {
                                id: 0,
                                dtype: Int,
                            },
                            Wire {
                                id: 2,
                                dtype: Int,
                            },
                            Wire {
                                id: 0,
                                dtype: Int,
                            },
                            Wire {
                                id: 1,
                                dtype: Int,
                            },
                            Wire {
                                id: 2,
                                dtype: Int,
                            },
                        ],
                    ],
                },
                write: Interface {
                    wires: [
                        [
                            Wire {
                                id: 10,
                                dtype: Bool,
                            },
                            Wire {
                                id: 10,
                                dtype: Bool,
                            },
                            Wire {
                                id: 11,
                                dtype: Bool,
                            },
                            Wire {
                                id: 11,
                                dtype: Bool,
                            },
                            Wire {
                                id: 12,
                                dtype: Bool,
                            },
                            Wire {
                                id: 12,
                                dtype: Bool,
                            },
                            Wire {
                                id: 13,
                                dtype: Int,
                            },
                            Wire {
                                id: 13,
                                dtype: Int,
                            },
                            Wire {
                                id: 14,
                                dtype: Int,
                            },
                            Wire {
                                id: 14,
                                dtype: Int,
                            },
                            Wire {
                                id: 15,
                                dtype: Int,
                            },
                            Wire {
                                id: 15,
                                dtype: Int,
                            },
                            Wire {
                                id: 5,
                                dtype: Int,
                            },
                            Wire {
                                id: 5,
                                dtype: Int,
                            },
                            Wire {
                                id: 6,
                                dtype: Int,
                            },
                            Wire {
                                id: 6,
                                dtype: Int,
                            },
                            Wire {
                                id: 7,
                                dtype: Int,
                            },
                            Wire {
                                id: 7,
                                dtype: Int,
                            },
                        ],
                    ],
                },
            },
        },
    ],
}