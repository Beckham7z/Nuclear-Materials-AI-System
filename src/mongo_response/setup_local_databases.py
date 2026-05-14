#!/usr/bin/env python3
"""
本地数据库设置脚本
由于网络限制，使用轻量级替代方案来建立三个数据库
"""

import os
import json
import sqlite3
import pickle
import time
from pathlib import Path
import logging

# 设置日志
