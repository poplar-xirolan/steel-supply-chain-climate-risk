import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_combination_declares_all_facility_identity_weight_inputs(tmp_path):
    from steel_supply.reproduction.plan import build_plan,external_inputs
    from steel_supply.reproduction.execution import fingerprint
    stage=build_plan(first='hazard-combine',last='hazard-combine')[0]
    inventories=[f'result/facilities/{name}_scaled.parquet' for name in ['mine','iron_factory','steel_factory']]
    assert set(inventories).issubset(external_inputs([stage]))
    for name in inventories:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    before=fingerprint(tmp_path,inventories)
    (tmp_path/inventories[0]).write_bytes(b'changed production weight')
    assert before!=fingerprint(tmp_path,inventories)


def test_mfa_stage_owns_correction_sidecar_even_before_parquet_exists(tmp_path):
    from steel_supply.reproduction.plan import build_plan
    from steel_supply.reproduction.execution import artifact_files
    stage=build_plan(first='mfa',last='mfa')[0]
    sidecar=tmp_path/'result/mfa/2023_m8_s1_asu_fsp_corrections.csv'
    sidecar.parent.mkdir(parents=True);sidecar.write_text('country,change\nAAA,1\n')
    assert sidecar in [p for selector in stage['outputs'] for p in artifact_files(tmp_path,selector)]


def test_mfa_optional_receipt_runs_without_git(tmp_path,monkeypatch):
    from steel_supply.pipeline_paths import bundle_root
    from steel_supply.pipeline_receipts import sha256_file,write_receipt
    from steel_supply.mfa.mfa_validation import validate_mfa_schema,calculate_node_balance
    from steel_supply.smoke import toy_network
    import subprocess
    builder=bundle_root()/'01_mfa/05_build_mfa.py'
    tree=ast.parse(builder.read_text(encoding='utf-8'))
    receipt_block=next(n for n in reversed(tree.body) if isinstance(n,ast.If))
    output=tmp_path/'network.parquet';toy_network().to_parquet(output,index=False)
    book=tmp_path/'source_book.xlsx';book.write_bytes(b'content fingerprint fixture')
    trade=tmp_path/'source_trade.parquet';trade.write_bytes(b'content fingerprint fixture')
    def no_git(*args,**kwargs):raise AssertionError('Standalone receipt must not invoke Git')
    monkeypatch.setattr(subprocess,'run',no_git)
    receipt=tmp_path/'receipt.json'
    scope=dict(args=SimpleNamespace(receipt_path=receipt,disable_m8_s1_asu_fsp_correction=False),
               df_result=toy_network(),DF_TRADE=pd.DataFrame(),YEAR=2023,
               validate_mfa_schema=validate_mfa_schema,calculate_node_balance=calculate_node_balance,
               OUTPUT_PARQUET_PATH=str(output),INPUT_WORKBOOK_PATH=str(book),TRADE_DATA_PATH=str(trade),
               bundle_root=bundle_root,subprocess=subprocess,Path=Path,__file__=str(builder),
               hashlib=hashlib,sha256_file=sha256_file,write_receipt=write_receipt,
               datetime=datetime,timezone=timezone,logger=SimpleNamespace(info=lambda *args:None))
    exec(compile(ast.Module(body=receipt_block.body,type_ignores=[]),str(builder),'exec'),scope)
    value=json.loads(receipt.read_text())
    assert value['status']=='PASS'
    assert value['code_commit']=='not-applicable-exported-source'
    assert value['input_hashes']['builder_source_sha256']==sha256_file(builder)


def test_statistics_source_paths_resolve_from_workspace(tmp_path,monkeypatch):
    from steel_supply.postprocessing import fig6_statistics as module
    a,b=tmp_path/'data/provided/input.csv',tmp_path/'result/new.csv'
    for path in [a,b]:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('x\n1\n')
    monkeypatch.setenv('STEEL_ANALYSE_ROOT',str(tmp_path))
    monkeypatch.setattr(module,'METRICS_SOURCE',a)
    monkeypatch.setattr(module,'SCORES_SOURCE',b)
    assert set(module.source_hashes())=={'data/provided/input.csv','result/new.csv'}
