from .backends.fsm import FSM
from .langs.toy_fsm import ToyFSM
from .nets.simple_lstm import SimpleLSTM

INDEX = {
    'backends' : {
        FSM.name: FSM,
    },
    'langs' : {
        ToyFSM.name: ToyFSM,
    },
    'nets' : {
        SimpleLSTM.name: SimpleLSTM,
    },
}
