# hailei_crew_setup.py - HumanLayer Integration for Human-in-the-Loop Workflow
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.tools import BaseTool, tool
try:
    from humanlayer import HumanLayer
except ImportError:
    # Mock HumanLayer for environments where it's not available
    class HumanLayer:
        def __init__(self, run_id=None):
            self.run_id = run_id
        
        def require_approval(self):
            def decorator(func):
                return lambda *args, **kwargs: func(*args, **kwargs)
            return decorator
        
        def human_as_tool(self):
            return lambda message: f"Human response to: {message}"
from typing import Type, Any
import yaml
import os
import json
from tools.blooms_taxonomy_tool import blooms_taxonomy_tool
from tools.accessibility_checker_tool import accessibility_checker_tool
from tools.resource_search_tool import resource_search_tool

@CrewBase
class HAILEICourseDesign:
    """HAILEI Course Design Crew with Educational Intelligence Agents"""

    agents_config = os.path.abspath("config/agents.yaml")
    tasks_config = os.path.abspath("config/tasks.yaml")

    def __init__(self):
        # Initialize HumanLayer for human-in-the-loop workflow
        self.hl = HumanLayer(run_id="hailei-course-design")
        
        # Load YAML configs
        with open(self.agents_config, 'r') as f:
            self._agents_dict = yaml.safe_load(f)
        with open(self.tasks_config, 'r') as f:
            self._tasks_dict = yaml.safe_load(f)

        print("Loaded HAILEI agents:", list(self._agents_dict.keys()))
        print("HumanLayer initialized for human approval workflow")

    # HumanLayer tools for approval workflow
    def create_approval_tools(self):
        """Create HumanLayer tools for human approval at each phase"""
        
        @tool("approve_phase_start")
        def approve_phase_start(phase_name: str, phase_description: str) -> str:
            """Request human approval to start a new phase"""
            # Create HumanLayer approval function
            approval_func = self.hl.require_approval()(
                lambda pn, pd: f"Human approved starting {pn}: {pd}"
            )
            return approval_func(phase_name, phase_description)

        @tool("approve_agent_output") 
        def approve_agent_output(agent_name: str, output_summary: str, full_output: str) -> str:
            """Request human approval of agent output before proceeding"""
            approval_func = self.hl.require_approval()(
                lambda an, os, fo: f"Human approved output from {an}: {os}"
            )
            return approval_func(agent_name, output_summary, full_output)

        @tool("approve_final_deliverable")
        def approve_final_deliverable(deliverable_summary: str) -> str:
            """Request human approval of final course design package"""
            approval_func = self.hl.require_approval()(
                lambda ds: f"Human approved final deliverable: {ds}"
            )
            return approval_func(deliverable_summary)

        @tool("human_communication")
        def human_communication(message: str) -> str:
            """Direct communication with human for questions or clarifications"""
            return self.hl.human_as_tool()(message)
        
        return [
            approve_phase_start,
            approve_agent_output, 
            approve_final_deliverable,
            human_communication
        ]

    @agent
    def hailei4t_coordinator_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["hailei4t_coordinator_agent"],
            verbose=True,
            memory=False,
            tools=[]  # Manager agent cannot have tools in hierarchical process
        )

    @agent
    def ipdai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["ipdai_agent"],
            verbose=True,
            memory=False,
            tools=[blooms_taxonomy_tool]
        )

    @agent
    def cauthai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["cauthai_agent"],
            verbose=True,
            memory=False,
            tools=[resource_search_tool]
        )

    @agent
    def tfdai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["tfdai_agent"],
            verbose=True,
            memory=False
        )

    @agent
    def editorai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["editorai_agent"],
            verbose=True,
            memory=False,
            tools=[accessibility_checker_tool, blooms_taxonomy_tool]
        )

    @agent
    def ethosai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["ethosai_agent"],
            verbose=True,
            memory=False,
            tools=[accessibility_checker_tool]
        )

    @agent
    def searchai_agent(self) -> Agent:
        return Agent(
            **self._agents_dict["searchai_agent"],
            verbose=True,
            memory=False,
            tools=[resource_search_tool]
        )

    @agent  
    def human_interface_agent(self) -> Agent:
        return Agent(
            role="Human Interface Agent",
            goal="Handle all human approvals and interactions using HumanLayer tools",
            backstory="You are responsible for managing human approvals at each phase of the HAILEI workflow. You present phase information to humans and collect their approval decisions using HumanLayer tools.",
            verbose=True,
            memory=False,
            tools=self.create_approval_tools(),
            allow_delegation=False
        )

    def create_master_coordination_task(self, course_request, kdka_framework, prrr_framework, lms_platform="Canvas"):
        """Single comprehensive task for coordinator to manage entire workflow"""
        
        task_description = f"""
        As the HAILEI Educational Intelligence Coordinator, you are the project manager overseeing the complete course design workflow. 
        You will coordinate with specialist agents step-by-step, presenting their work to the human for approval at each stage.

        COURSE REQUEST TO PROCESS:
        {json.dumps(course_request, indent=2)}

        FRAMEWORK SPECIFICATIONS:
        KDKA Model: {json.dumps(kdka_framework, indent=2)}
        PRRR Framework: {json.dumps(prrr_framework, indent=2)}
        Target LMS: {lms_platform}

        CRITICAL DELEGATION INSTRUCTIONS:
        When using the "Delegate work to coworker" tool, you MUST provide simple string values, NOT dictionaries.
        
        Example CORRECT format:
        - task: "Create course foundation using KDKA and PRRR frameworks for Introduction to Artificial Intelligence course"
        - context: "The course is 16 weeks, 3 credits, undergraduate level covering AI foundations, history, and applications"
        - coworker: "Instructional Planning & Design Agent (IPDAi)"

        Example WRONG format (DO NOT USE):
        - task: {{"description": "Create foundation", "type": "str"}}

        YOUR WORKFLOW MANAGEMENT RESPONSIBILITIES:

        **CRITICAL WORKFLOW PROTOCOL:**
        1. Delegate to "Human Interface Agent" to present phase overview and get approval
        2. Get explicit human approval before proceeding
        3. Delegate work to specialist agent  
        4. Delegate to "Human Interface Agent" to present results and get approval
        5. Get human approval/revision requests before proceeding
        6. Only proceed to next phase after human approval
        7. Use Human Interface Agent for all human interactions

        **PHASE-BY-PHASE EXECUTION:**

        1. **START PHASE**: 
           - Delegate to "Human Interface Agent" with task: "Get human approval to start course design workflow"
           - Context: "Use approve_phase_start tool with phase_name='Course Overview' and phase_description='Begin HAILEI course design workflow for Introduction to Artificial Intelligence with 5 specialist phases'"
           - Wait for human approval before proceeding

        2. **PHASE 1 - Course Foundation (IPDAi)**:
           - Delegate to "Human Interface Agent" with task: "Get human approval for Phase 1"
           - Context: "Use approve_phase_start tool with phase_name='Phase 1 - Course Foundation' and phase_description='Delegate course foundation creation to IPDAi specialist using KDKA and PRRR frameworks'"
           - If approved, delegate to "Instructional Planning & Design Agent (IPDAi)" with task:
             "Create comprehensive course foundation for {course_request.get('course_title', 'the course')} using KDKA and PRRR frameworks. Generate learning objectives validated with Bloom's taxonomy, structure weekly module breakdown for {course_request.get('course_duration_weeks', 16)} weeks, and draft initial syllabus."
           - Context: "Course details: {course_request.get('course_title', '')} - {course_request.get('course_description', '')[:100]}... Level: {course_request.get('course_level', '')}. Use provided KDKA and PRRR frameworks for alignment."
           - Delegate to "Human Interface Agent" with task: "Get human approval for IPDAi output"
           - Context: "Use approve_agent_output tool with agent_name='IPDAi', output_summary='Course foundation with learning objectives and syllabus', full_output='[complete IPDAi output]'"
           - Only proceed after human approval

        3. **PHASE 2 - Content Creation (CAuthAi)**:
           - Use "approve_phase_start" tool with phase_name="Phase 2 - Content Creation" and phase_description="Delegate instructional content creation to CAuthAi specialist using PRRR methodology"
           - If approved, delegate to "Course Authoring Agent (CAuthAi)" with task:
             "Develop comprehensive instructional content based on the approved course foundation. Generate PRRR-based learning activities, create assessments and rubrics, and curate educational resources."
           - Context: "Use the course foundation created by IPDAi and apply PRRR methodology for engagement."
           - Use "approve_agent_output" tool with agent_name="CAuthAi", output_summary="Instructional content with PRRR-based activities", full_output="[complete CAuthAi output]"
           - Only proceed after human approval

        4. **PHASE 3 - Technical Design (TFDAi)**:
           - Use "approve_phase_start" tool with phase_name="Phase 3 - Technical Design" and phase_description="Delegate LMS technical design to TFDAi specialist for Canvas platform"
           - If approved, delegate to "Technical & Functional Design Agent (TFDAi)" with task:
             "Create comprehensive LMS implementation plan for {lms_platform} platform. Map educational content to platform features and design technical specifications."
           - Context: "Use the instructional content created by CAuthAi to design technical implementation."
           - Use "approve_agent_output" tool with agent_name="TFDAi", output_summary="LMS technical implementation plan", full_output="[complete TFDAi output]"
           - Only proceed after human approval

        5. **PHASE 4 - Quality Review (EditorAi)**:
           - Use "approve_phase_start" tool with phase_name="Phase 4 - Quality Review" and phase_description="Delegate quality review and accessibility compliance to EditorAi specialist"
           - If approved, delegate to "Content Review & Enhancement Agent (EditorAi)" with task:
             "Review and enhance all course content for quality, accessibility compliance, and framework alignment. Validate using accessibility and Bloom's taxonomy tools."
           - Context: "Review all outputs from IPDAi, CAuthAi, and TFDAi for consistency and compliance."
           - Use "approve_agent_output" tool with agent_name="EditorAi", output_summary="Quality-reviewed and accessibility-compliant content", full_output="[complete EditorAi output]"
           - Only proceed after human approval

        6. **PHASE 5 - Ethical Compliance (EthosAi)**:
           - Use "approve_phase_start" tool with phase_name="Phase 5 - Ethical Compliance" and phase_description="Delegate ethical audit and UDL validation to EthosAi specialist"
           - If approved, delegate to "Ethical Oversight Agent (EthosAi)" with task:
             "Conduct comprehensive ethical compliance audit of the complete course package. Validate UDL principles and ensure inclusive content using accessibility tools."
           - Context: "Audit the final reviewed content from EditorAi for ethical compliance and inclusivity."
           - Use "approve_agent_output" tool with agent_name="EthosAi", output_summary="Ethical compliance audit and UDL validation", full_output="[complete EthosAi output]"

        7. **FINAL INTEGRATION**:
           - Compile all approved outputs into comprehensive course package
           - Provide implementation roadmap
           - Use "approve_final_deliverable" tool with deliverable_summary="Complete HAILEI course design package with all approved components including foundation, content, technical design, quality review, and ethical compliance"

        **IMPORTANT DELEGATION NOTES:**
        - Use exact agent role names for delegation, not technical IDs
        - Always use HumanLayer approval tools before proceeding to next phase
        - Use "approve_phase_start" before each delegation
        - Use "approve_agent_output" after each specialist completes work
        - Use "approve_final_deliverable" for the complete course package
        - Wait for explicit human approval before each delegation
        - Handle revision requests by re-delegating with specific changes
        
        **HUMANLAYER TOOLS AVAILABLE:**
        - approve_phase_start: Get approval before starting each phase
        - approve_agent_output: Get approval of specialist outputs
        - approve_final_deliverable: Get approval of final deliverable
        - human_communication: Direct human communication when needed
        """

        return Task(
            description=task_description,
            agent=self.hailei4t_coordinator_agent(),
            expected_output="""
            Complete HAILEI course design package including:
            - Human-approved course foundation (syllabus, objectives, structure)
            - Human-approved instructional content (activities, assessments, resources)  
            - Human-approved technical implementation plan
            - Human-approved quality-reviewed materials
            - Human-approved ethical compliance certification
            - Final integrated course package with implementation roadmap
            - Confirmation of human approval for final deliverable
            """,
            human_input=False,  # HumanLayer handles human interaction
            tools=[]  # Coordinator uses delegation and HumanLayer approval tools automatically
        )

    def run_course_design(self, course_request, kdka_framework, prrr_framework, lms_platform="Canvas"):
        """Run the complete HAILEI course design workflow with coordinator management"""
        print(f"[INFO] Starting HAILEI hierarchical course design for: {course_request.get('course_title', 'Untitled Course')}")
        print("[INFO] Coordinator agent will manage all specialists with human approval at each stage")
        
        # Create single master task for coordinator
        master_task = self.create_master_coordination_task(course_request, kdka_framework, prrr_framework, lms_platform)

        # Create hierarchical crew with coordinator managing specialists
        # IMPORTANT: Manager agent should NOT be in agents list for hierarchical process
        crew = Crew(
            agents=[
                # Only specialist agents - manager is separate
                self.human_interface_agent(),  # Add human interface agent with HumanLayer tools
                self.ipdai_agent(),
                self.cauthai_agent(),
                self.tfdai_agent(),
                self.editorai_agent(),
                self.ethosai_agent(),
                self.searchai_agent()
            ],
            tasks=[master_task],  # Single task for coordinator to manage workflow
            process=Process.hierarchical,
            manager_agent=self.hailei4t_coordinator_agent(),  # Coordinator as project manager
            verbose=True,
            memory=False,
            max_rpm=10,
            max_retry_limit=2
        )

        return crew.kickoff()

    @crew
    def run(self) -> Crew:
        """For compatibility with existing crew framework"""
        return Crew(
            agents=[
                # Only specialist agents - manager is separate in hierarchical mode
                self.human_interface_agent(),  # Add human interface agent with HumanLayer tools
                self.ipdai_agent(),
                self.cauthai_agent(),
                self.tfdai_agent(),
                self.editorai_agent(),
                self.ethosai_agent(),
                self.searchai_agent()
            ],
            tasks=[],  # Tasks created dynamically by coordinator
            process=Process.hierarchical,
            manager_agent=self.hailei4t_coordinator_agent(),
            verbose=True,
            memory=False,
            max_rpm=10,
            max_retry_limit=2
        )