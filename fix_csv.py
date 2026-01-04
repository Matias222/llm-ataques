import re


file_path = 'data/advbench/harmful_behaviors.csv'
output_path= 'data/advbench/harmful_behaviors_fixed.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace comma with semicolon only when it appears before a quote
# This regex finds: comma followed by optional whitespace followed by a quote
fixed_content = re.sub(r',(\s*")', r';\1', content)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed CSV saved to {output_path}")
print(f"Original: {content}")
print(f"Fixed: {fixed_content}")
