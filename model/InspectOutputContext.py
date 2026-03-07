
import traceback
import torch.nn as nn
from functools import partial
from torch.utils.hooks import RemovableHandle


class InspectOutputContext:
    def __init__(self, model, module_names, move_to_cpu=False, last_position=False, save_generation=False,
                 save_dir=None, track_grads=False):
        self.model = model
        self.module_names = module_names
        self.move_to_cpu = move_to_cpu
        self.last_position = last_position
        self.save_generation = save_generation
        self.save_dir = save_dir
        self.track_grads = track_grads  # NUOVO PARAMETRO
        self.handles = []
        self.catcher = dict()
        self.final_output = None

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.module_names:
                handle = self.inspect_output(module, self.catcher, module_name, move_to_cpu=self.move_to_cpu,
                                             last_position=self.last_position)
                self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            traceback.print_tb(exc_tb)
            return False
        return True

    def inspect_output(self, module: nn.Module, catcher: dict, module_name, move_to_cpu,
                       last_position=False) -> RemovableHandle:
        hook_instance = partial(self.inspect_hook, catcher=catcher, module_name=module_name, move_to_cpu=move_to_cpu,
                                last_position=last_position)
        handle = module.register_forward_hook(hook_instance)
        return handle

    def inspect_hook(self, module: nn.Module, inputs, outputs, catcher: dict, module_name, move_to_cpu,
                     last_position=False):
        # 1. Estrarre il tensore principale (spesso HuggingFace restituisce tuple)
        tensor = outputs[0] if type(outputs) is tuple else outputs

        # 2. Se vogliamo tracciare i gradienti, abilitiamo requires_grad e retain_grad
        if self.track_grads:
            if not tensor.requires_grad:
                tensor.requires_grad_(True)
            tensor.retain_grad()

            # Salviamo il tensore intero nel catcher (faremo slicing DOPO il backward)
            catcher[module_name] = tensor
        else:
            # Comportamento originale
            if last_position:
                catcher[module_name] = tensor[:, -1]
            else:
                catcher[module_name] = tensor
            if move_to_cpu:
                catcher[module_name] = catcher[module_name].cpu()

        return outputs
