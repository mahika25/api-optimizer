import ast
import os
from typing import List, Dict, Set
from pathlib import Path

class APICallVisitor(ast.NodeVisitor):
 
    def __init__(self):
        self.calls = []
        self.current_file = None
    
    def visit_Call(self, node):
        # Detect: client.chat.completions.create(...)
        # or: openai.ChatCompletion.create(...)
        
        if self._is_api_call(node):
            call_info = self._extract_call_info(node)
            if call_info:
                call_info['file'] = self.current_file
                call_info['line'] = node.lineno
                self.calls.append(call_info)
        
        self.generic_visit(node)
    
    def _is_api_call(self, node) -> bool:
        try:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'create':
                    current = node.func.value
                    attrs = []
                    
                    while isinstance(current, ast.Attribute):
                        attrs.append(current.attr)
                        current = current.value
                    
                    attrs.reverse()
                    
                    if 'completions' in attrs or 'ChatCompletion' in attrs:
                        return True
            
            return False
        except:
            return False
    
    def _extract_call_info(self, node) -> Dict:
        info = {
            'model': None,
            'prompt': None,
            'prompt_pattern': None,
            'temperature': None,
            'max_tokens': None
        }
        
        for keyword in node.keywords:
            arg_name = keyword.arg
            
            if arg_name == 'model':
                info['model'] = self._extract_value(keyword.value)
            
            elif arg_name == 'messages':
                info['prompt'], info['prompt_pattern'] = self._extract_messages(keyword.value)
            
            elif arg_name == 'temperature':
                info['temperature'] = self._extract_value(keyword.value)
            
            elif arg_name == 'max_tokens':
                info['max_tokens'] = self._extract_value(keyword.value)
        
        return info if info['model'] or info['prompt'] else None
    
    def _extract_value(self, node) -> any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            return f"<variable: {node.id}>"
        elif isinstance(node, ast.JoinedStr):
            return "<f-string>"
        else:
            return "<dynamic>"
    
    def _extract_messages(self, node):
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Dict):
                    for key, value in zip(elt.keys, elt.values):
                        if isinstance(key, ast.Constant) and key.value == 'content':
                            content = self._extract_value(value)
                            pattern = self._detect_pattern(value)
                            return content, pattern
        
        return None, None
    
    def _detect_pattern(self, node) -> str:
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    parts.append("{...}")
            return "".join(parts)
        
        return None


class StaticAnalyzer:
    
    def __init__(self):
        self.visitor = APICallVisitor()
    
    def analyze_file(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code, filename=filepath)
            self.visitor.current_file = filepath
            self.visitor.visit(tree)
            
            return self.visitor.calls
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return []
    
    def analyze_directory(self, directory: str) -> List[Dict]:
        all_calls = []
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    calls = self.analyze_file(filepath)
                    all_calls.extend(calls)
        
        return all_calls
    
    def generate_report(self, calls: List[Dict]):
        print("\n" + "="*60)
        print("STATIC ANALYSIS REPORT")
        print("="*60)
        print(f"Total API calls found: {len(calls)}\n")
        
        by_model = {}
        for call in calls:
            model = call.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(call)
        
        print("Calls by Model:")
        for model, model_calls in sorted(by_model.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {model}: {len(model_calls)} calls")
        
        # Find patterns
        print("\n Prompt Patterns:")
        patterns = {}
        for call in calls:
            pattern = call.get('prompt_pattern') or call.get('prompt')
            if pattern and pattern != '<dynamic>':
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(call)
        
        for pattern, pattern_calls in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
            if len(pattern_calls) > 1:
                print(f"  \"{pattern[:60]}...\" appears {len(pattern_calls)} times")
                print(f"    Files: {', '.join(set(c['file'] for c in pattern_calls))}")
        
        # Optimization suggestions
        from src.analyzer.model_rec import ModelRecommender
        recommender = ModelRecommender()
        
        print("\n💡 Optimization Opportunities:")
        suggestions = 0
        
        for call in calls:
            model = call.get('model')
            prompt = call.get('prompt')
            
            if model and prompt and prompt not in ['<dynamic>', '<f-string>']:
                rec = recommender.recommend_model(model, prompt)
                
                if rec:
                    suggestions += 1
                    print(f"\n  {call['file']}:{call['line']}")
                    print(f"    Current: {model}")
                    print(f"    Suggested: {rec['recommended_model']}")
                    print(f"    Reason: {rec['reason']}")
                    print(f"    Savings: ${rec['cost_savings_per_1m_tokens']:.2f}/1M tokens")
        
        if suggestions == 0:
            print("  No immediate optimizations found")
        
        print("\n" + "="*60)