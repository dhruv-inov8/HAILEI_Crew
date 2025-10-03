# tools/accessibility_checker_tool.py
from crewai.tools import tool
import re

@tool("Accessibility Compliance Checker")
def accessibility_checker_tool(content: str, content_type: str = "text", check_level: str = "AA") -> str:
    """
    Validates educational content against WCAG guidelines and accessibility standards. 
    Checks for compliance with Universal Design for Learning (UDL) principles and 
    provides specific recommendations for improvement.
    
    Args:
        content: Educational content to check for accessibility compliance
        content_type: Type of content: text, html, markdown, or mixed
        check_level: WCAG compliance level: A, AA, or AAA
    """
    
    def check_text_content(text):
        issues = []
        suggestions = []
        
        # Check for image references without alt text descriptions
        img_pattern = r'!\[([^\]]*)\]\([^)]+\)'
        images = re.findall(img_pattern, text)
        for alt_text in images:
            if not alt_text.strip():
                issues.append("Images found without descriptive alt text")
                suggestions.append("Add descriptive alt text for all images (avoid 'image of' or 'picture of')")
        
        # Check for color-only information conveyance
        color_indicators = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'see the red', 'green indicates', 'blue shows']
        color_references = [indicator for indicator in color_indicators if indicator in text.lower()]
        if color_references:
            issues.append("Possible reliance on color alone to convey information")
            suggestions.append("Supplement color coding with text labels, symbols, or patterns")
        
        # Check reading level complexity
        sentences = text.split('.')
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if len(long_sentences) > len(sentences) * 0.3:
            issues.append("High proportion of complex sentences may impact readability")
            suggestions.append("Break down complex sentences for better comprehension")
        
        # Check for jargon without explanation
        complex_terms = ['algorithm', 'paradigm', 'methodology', 'framework', 'infrastructure', 'optimization']
        unexplained_jargon = [term for term in complex_terms if term in text.lower() and f"({term}" not in text.lower()]
        if unexplained_jargon:
            issues.append(f"Technical terms may need explanation: {', '.join(unexplained_jargon)}")
            suggestions.append("Define technical terms or provide glossary links")
        
        # Check for clear headings structure
        heading_pattern = r'^#+\s'
        headings = re.findall(heading_pattern, text, re.MULTILINE)
        if len(text.split('\n')) > 20 and len(headings) < 3:
            issues.append("Long content lacks clear heading structure")
            suggestions.append("Add descriptive headings to organize content")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "score": max(0, 100 - len(issues) * 15)
        }
    
    def check_udl_compliance(text):
        udl_principles = {
            "multiple_means_representation": {
                "description": "Provide multiple ways of presenting information",
            },
            "multiple_means_engagement": {
                "description": "Provide multiple ways to motivate learners",
            },
            "multiple_means_action_expression": {
                "description": "Provide multiple ways for learners to express knowledge",
            }
        }
        
        udl_analysis = {}
        
        for principle, data in udl_principles.items():
            compliance_score = 0
            recommendations = []
            
            if principle == "multiple_means_representation":
                # Check for multiple formats
                has_text = len(text) > 100
                has_structure = '##' in text or '**' in text
                has_examples = 'example' in text.lower() or 'for instance' in text.lower()
                
                compliance_score = sum([has_text, has_structure, has_examples]) / 3 * 100
                
                if not has_examples:
                    recommendations.append("Add concrete examples to illustrate concepts")
                if not has_structure:
                    recommendations.append("Use formatting to highlight key information")
                    
            elif principle == "multiple_means_engagement":
                # Check for engagement elements
                has_questions = '?' in text
                has_activities = any(word in text.lower() for word in ['activity', 'exercise', 'practice', 'try'])
                has_relevance = any(word in text.lower() for word in ['why', 'because', 'important', 'relevant'])
                
                compliance_score = sum([has_questions, has_activities, has_relevance]) / 3 * 100
                
                if not has_questions:
                    recommendations.append("Include reflection questions to engage learners")
                if not has_activities:
                    recommendations.append("Add interactive elements or practice opportunities")
                    
            elif principle == "multiple_means_action_expression":
                # Check for expression options
                has_choices = any(word in text.lower() for word in ['choose', 'select', 'option', 'alternative'])
                has_formats = any(word in text.lower() for word in ['write', 'present', 'demonstrate', 'create'])
                has_scaffolding = any(word in text.lower() for word in ['step', 'guide', 'template', 'framework'])
                
                compliance_score = sum([has_choices, has_formats, has_scaffolding]) / 3 * 100
                
                if not has_choices:
                    recommendations.append("Provide multiple ways for students to engage with content")
                if not has_scaffolding:
                    recommendations.append("Include step-by-step guidance or templates")
            
            udl_analysis[principle] = {
                "score": compliance_score,
                "recommendations": recommendations,
                "description": data["description"]
            }
        
        return udl_analysis
    
    # Perform analysis
    text_analysis = check_text_content(content)
    udl_analysis = check_udl_compliance(content)
    
    # Generate report
    report = "# Accessibility Compliance Report\n\n"
    
    # Overall score calculation
    text_score = text_analysis["score"]
    udl_scores = [data["score"] for data in udl_analysis.values()]
    overall_score = (text_score + sum(udl_scores) / len(udl_scores)) / 2
    
    report += f"**Overall Accessibility Score: {overall_score:.1f}/100**\n"
    report += f"**WCAG Level Target: {check_level}**\n\n"
    
    # Text accessibility analysis
    report += "## WCAG Compliance Analysis\n\n"
    if text_analysis["issues"]:
        report += "### Issues Identified:\n"
        for issue in text_analysis["issues"]:
            report += f"- {issue}\n"
        report += "\n"
    
    if text_analysis["suggestions"]:
        report += "### Recommendations:\n"
        for suggestion in text_analysis["suggestions"]:
            report += f"- {suggestion}\n"
        report += "\n"
    else:
        report += "No major WCAG compliance issues detected.\n\n"
    
    # UDL analysis
    report += "## Universal Design for Learning (UDL) Analysis\n\n"
    for principle, data in udl_analysis.items():
        principle_name = principle.replace('_', ' ').title()
        report += f"### {principle_name}\n"
        report += f"**Score: {data['score']:.1f}/100**\n"
        report += f"*{data['description']}*\n\n"
        
        if data["recommendations"]:
            report += "**Recommendations:**\n"
            for rec in data["recommendations"]:
                report += f"- {rec}\n"
            report += "\n"
        else:
            report += "Meets UDL guidelines for this principle.\n\n"
    
    # Priority action items
    all_suggestions = text_analysis["suggestions"] + [rec for data in udl_analysis.values() for rec in data["recommendations"]]
    if all_suggestions:
        report += "## Priority Action Items\n\n"
        priority_items = all_suggestions[:5]  # Top 5 recommendations
        for i, item in enumerate(priority_items, 1):
            report += f"{i}. {item}\n"
        report += "\n"
    
    # Compliance status
    if overall_score >= 90:
        status = "Excellent accessibility compliance"
    elif overall_score >= 75:
        status = "Good accessibility with minor improvements needed"
    elif overall_score >= 60:
        status = "Moderate accessibility issues require attention"
    else:
        status = "Significant accessibility improvements required"
    
    report += f"## Compliance Status\n\n**{status}**\n\n"
    
    # Additional resources
    report += "## Additional Resources\n\n"
    report += "- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)\n"
    report += "- [UDL Guidelines](http://udlguidelines.cast.org/)\n"
    report += "- [WebAIM Accessibility Checklist](https://webaim.org/standards/wcag/checklist)\n"
    
    return report