# tools/blooms_taxonomy_tool.py
from crewai.tools import tool

@tool("Blooms Taxonomy Validator")
def blooms_taxonomy_tool(content: str, target_level: str = "", course_level: str = "undergraduate") -> str:
    """
    Validates learning objectives against Bloom's Taxonomy levels and generates appropriate 
    action verbs for educational content. Ensures cognitive complexity matches course level 
    and provides suggestions for improvement.
    
    Args:
        content: Learning objective or educational content to analyze
        target_level: Target Bloom's level (Remember, Understand, Apply, Analyze, Evaluate, Create)
        course_level: Course level for appropriate cognitive complexity
    """
    
    blooms_levels = {
        "remember": {
            "description": "Recall facts and basic concepts",
            "verbs": ["define", "describe", "identify", "know", "label", "list", "match", "name", "outline", "recall", "recognize", "reproduce", "select", "state"],
            "complexity": 1
        },
        "understand": {
            "description": "Explain ideas or concepts", 
            "verbs": ["classify", "compare", "contrast", "demonstrate", "explain", "extend", "illustrate", "infer", "interpret", "outline", "relate", "rephrase", "show", "summarize", "translate"],
            "complexity": 2
        },
        "apply": {
            "description": "Use information in new situations",
            "verbs": ["apply", "build", "choose", "construct", "develop", "experiment", "identify", "interview", "make use of", "model", "organize", "plan", "select", "solve", "utilize"],
            "complexity": 3
        },
        "analyze": {
            "description": "Draw connections among ideas",
            "verbs": ["analyze", "break down", "compare", "contrast", "diagram", "deconstruct", "differentiate", "discriminate", "distinguish", "examine", "experiment", "identify", "illustrate", "infer", "outline", "relate", "select", "separate"],
            "complexity": 4
        },
        "evaluate": {
            "description": "Justify a stand or decision",
            "verbs": ["appraise", "argue", "assess", "attach", "choose", "compare", "defend", "estimate", "evaluate", "judge", "predict", "rate", "score", "select", "support", "value"],
            "complexity": 5
        },
        "create": {
            "description": "Produce new or original work",
            "verbs": ["assemble", "build", "collect", "combine", "compile", "compose", "construct", "create", "design", "develop", "formulate", "manage", "organize", "plan", "prepare", "propose", "set up", "write"],
            "complexity": 6
        }
    }
    
    def identify_bloom_level(content_text):
        content_lower = content_text.lower()
        detected_levels = []
        
        for level, data in blooms_levels.items():
            for verb in data["verbs"]:
                if verb in content_lower:
                    detected_levels.append({
                        "level": level,
                        "verb": verb,
                        "complexity": data["complexity"],
                        "description": data["description"]
                    })
        
        if detected_levels:
            return max(detected_levels, key=lambda x: x["complexity"])
        else:
            return {"level": "unidentified", "complexity": 0}
    
    def validate_course_level_alignment(bloom_level, course_lvl):
        bloom_complexity = blooms_levels.get(bloom_level, {}).get("complexity", 0)
        
        recommendations = {
            "introductory": {"min": 1, "max": 3, "focus": ["remember", "understand", "apply"]},
            "intermediate": {"min": 2, "max": 5, "focus": ["understand", "apply", "analyze", "evaluate"]},
            "advanced": {"min": 3, "max": 6, "focus": ["apply", "analyze", "evaluate", "create"]},
            "graduate": {"min": 4, "max": 6, "focus": ["analyze", "evaluate", "create"]}
        }
        
        course_level_lower = course_lvl.lower()
        if "introductory" in course_level_lower or "beginner" in course_level_lower:
            rec = recommendations["introductory"]
        elif "intermediate" in course_level_lower:
            rec = recommendations["intermediate"]
        elif "advanced" in course_level_lower:
            rec = recommendations["advanced"]
        elif "graduate" in course_level_lower or "master" in course_level_lower:
            rec = recommendations["graduate"]
        else:
            rec = recommendations["intermediate"]
        
        is_appropriate = rec["min"] <= bloom_complexity <= rec["max"]
        
        return {
            "is_appropriate": is_appropriate,
            "recommended_range": f"{rec['min']}-{rec['max']}",
            "recommended_levels": rec["focus"],
            "current_complexity": bloom_complexity
        }
    
    # Analyze content
    current_analysis = identify_bloom_level(content)
    
    # Validate alignment
    if current_analysis["level"] != "unidentified":
        alignment = validate_course_level_alignment(current_analysis["level"], course_level)
    else:
        alignment = {"is_appropriate": False, "current_complexity": 0}
    
    # Format response
    result = "**Bloom's Taxonomy Analysis**\n\n"
    
    if current_analysis["level"] != "unidentified":
        result += f"**Detected Level:** {current_analysis['level'].title()}\n"
        result += f"**Cognitive Focus:** {current_analysis['description']}\n"
        result += f"**Complexity:** {current_analysis['complexity']}/6\n"
        result += f"**Key Verb Found:** {current_analysis.get('verb', 'N/A')}\n\n"
        
        result += f"**Course Level Alignment:** {'✅ Appropriate' if alignment['is_appropriate'] else '⚠️ Needs Adjustment'}\n"
        result += f"**Recommended Range:** Complexity {alignment['recommended_range']}\n"
        result += f"**Suggested Levels for {course_level}:** {', '.join([l.title() for l in alignment['recommended_levels']])}\n\n"
    else:
        result += "**Detected Level:** No clear Bloom's taxonomy verbs identified\n"
        result += "**Recommendation:** Add specific action verbs to clarify learning expectations\n\n"
    
    if target_level and target_level.lower() in blooms_levels:
        level_data = blooms_levels[target_level.lower()]
        verbs = level_data["verbs"][:5]
        result += "**Suggestions for Target Level:**\n"
        result += f"1. Use action verbs from the '{target_level.title()}' level: {', '.join(verbs)}\n"
        result += f"2. Cognitive focus: {level_data['description']}\n"
        result += f"3. Example objective: 'Students will be able to {verbs[0]} [content] in order to [purpose]'\n\n"
    
    result += "**General Recommendations:**\n"
    result += "• Use specific, measurable action verbs\n"
    result += "• Align cognitive complexity with course level\n"
    result += "• Include context and purpose in objectives\n"
    result += "• Consider prerequisite knowledge and skills\n"
    
    return result