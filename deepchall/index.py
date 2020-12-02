from .backends.fsm import FSM
from .backends.cfg import CFG
from .langs.toy_fsm import ToyFSM
from .langs.toy_cfg import ToyCFG
from .nets.simple_lstm import SimpleLSTM

INDEX = {
    'backends' : {
        FSM.name: FSM,
        CFG.name: CFG,
    },
    'langs' : {
        ToyFSM.name: ToyFSM,
        ToyCFG.name: ToyCFG,
    },
    'nets' : {
        SimpleLSTM.name: SimpleLSTM,
    },
}
