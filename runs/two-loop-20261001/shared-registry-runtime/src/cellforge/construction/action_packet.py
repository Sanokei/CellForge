"""Mechanical typed validation of model-authored file action arguments.

No action selection, argument inference, shell execution or user-file writes.
Main owns authoring/admission; router owns numerical action selection.
"""
import copy
from pathlib import Path


def validate_file_action(packet, *, allowed_root, max_argument_bytes=4096):
    """Return an explicit checked target for a declared allowed fixture root.

    Expected packet is the existing call/name/arguments transport. A caller
    must authorize the root and execute the literal arguments independently.
    This validator grants no permission to mutate arbitrary user files.
    """
    if not isinstance(packet,dict) or set(packet)!={'type','name','arguments'} or packet['type']!='call':
        raise ValueError('exact call/name/arguments action packet required')
    name=packet['name'];args=packet['arguments']
    if name not in ('write_file','edit_in_place') or not isinstance(args,dict):
        raise ValueError('declared write_file or edit_in_place action required')
    required={'path','content'} if name=='write_file' else {'path','old','new','expected_matches'}
    if set(args)!=required:
        raise ValueError('exact model-authored argument fields required')
    path=args['path']
    if (not isinstance(path,str) or not path or len(path)>128
            or any(ord(c)<32 for c in path) or any(c in '<>:"|?*' for c in path)):
        raise ValueError('bounded relative Windows-safe path required')
    relative=Path(path)
    if relative.is_absolute() or relative.anchor or '..' in relative.parts:
        raise ValueError('absolute paths and traversal forbidden')
    reserved={'CON','PRN','AUX','NUL'}|{f'COM{i}' for i in range(1,10)}|{f'LPT{i}' for i in range(1,10)}
    if any(part in ('.','') or part.endswith((' ','.')) or part.split('.')[0].upper() in reserved for part in relative.parts):
        raise ValueError('ambiguous or device path forbidden')
    root=Path(allowed_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('explicit existing allowed fixture directory required')
    target=(root/relative).resolve()
    if target==root or not target.is_relative_to(root) or not target.parent.is_dir():
        raise ValueError('resolved target must remain inside allowed root')
    if target.exists() and not target.is_file():
        raise ValueError('regular-file target required')
    if name=='edit_in_place' and not target.is_file():
        raise ValueError('edit requires an existing fixture file')
    for key in ('content',) if name=='write_file' else ('old','new'):
        text=args[key]
        if not isinstance(text,str) or len(text.encode('utf-8'))>max_argument_bytes:
            raise ValueError('bounded model-authored UTF8 text required')
    if name=='edit_in_place' and (not args['old'] or type(args['expected_matches']) is not int or not 1<=args['expected_matches']<=16):
        raise ValueError('nonempty exact old text and expected_matches1..16 required')
    return {'packet':copy.deepcopy(packet),'resolved_path':str(target),
            'allowed_root':str(root),'encoding':'utf-8','execution_authority':'caller must authorize this root; validation only'}
