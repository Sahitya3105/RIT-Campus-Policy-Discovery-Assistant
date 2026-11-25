"""
mcp_server.py — FastMCP Tool Server for MSRIT Compliance & Academic Actions
===========================================================================
Exposes genuine compliance tools via stdio:
  1. check_student_eligibility: Verifies student attendance, condonation eligibility,
     and vertical progression backlogs against official MSRIT Autonomous Regulations.
  2. draft_official_petition: Auto-drafts formal, clause-cited petitions to the
     Principal / Head of Department.
"""

from fastmcp import FastMCP
import json
import os
import datetime
from pathlib import Path

mcp = FastMCP("MSRIT Compliance Tools")

STUDENTS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "students_db.json"


def load_students_db() -> dict:
    with open(STUDENTS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["students"]


@mcp.tool()
def check_student_eligibility(
    usn: str,
    course_code: str = "ALL",
    leave_type: str = "general"
) -> dict:
    """
    Check if a student is eligible for leave, exam clearance, or condonation
    under MSRIT Regulations (Clause 10.1, Clause 10.3, Addendum Clause 21.2).

    Args:
        usn: Student University Serial Number (e.g., "1MS22CS045")
        course_code: Specific course code or "ALL" for overall attendance
        leave_type: Type of leave: "cultural", "medical", "sports", "technical", "general"

    Returns:
        Eligibility status, attendance details, and applicable policy clauses.
    """
    try:
        students = load_students_db()
    except Exception as e:
        return {"error": f"Failed to load students database: {str(e)}"}

    # Case-insensitive USN lookup
    student = None
    usn_upper = usn.strip().upper()
    student_list = list(students.values()) if isinstance(students, dict) else students
    for s in student_list:
        if s.get("usn", "").upper() == usn_upper:
            student = s
            break

    if student is None:
        return {
            "error": f"Student with USN '{usn}' not found in database.",
            "usn": usn,
        }

    # Determine attendance percentage
    if course_code != "ALL" and "courses" in student:
        course_data = student.get("courses", {}).get(course_code)
        if course_data:
            attendance_pct = float(course_data.get("attendance", student.get("attendance_pct", 0)))
        else:
            attendance_pct = float(student.get("attendance_pct", 0))
    else:
        attendance_pct = float(student.get("attendance_pct", 0))

    backlogs = int(student.get("backlogs", 0))
    backlog_credits = int(student.get("backlog_credits", 0))
    cgpa = float(student.get("cgpa", 0.0))

    # Evaluate against MSRIT Clause 10.1 (85% Standard & 10% Principal Condonation Window)
    # Thresholds:
    # >= 85.0%: Eligible without condonation
    # 75.0% - 84.99%: Eligible with Principal Condonation (Medical, Cultural, Sports, Paper Presentation)
    # < 75.0%: Ineligible (Results in 'DX' grade under Clause 10.3 / Addendum Clause 21.2)
    if attendance_pct >= 85.0:
        eligible = True
        status = "ELIGIBLE"
        condonation_required = False
        attendance_deficit = 0.0
        applicable_clause = "Clause 10.1 (85% Minimum Attendance Satisfied)"
        recommendation = "Student satisfies standard autonomous attendance criteria. Approved for regular SEE appearance."
    elif 75.0 <= attendance_pct < 85.0:
        eligible = True
        status = "ELIGIBLE_WITH_CONDONATION"
        condonation_required = True
        attendance_deficit = round(85.0 - attendance_pct, 1)
        applicable_clause = "Clause 10.1 (Principal 10% Condonation Window — Effective Floor 75%)"
        recommendation = (
            f"Attendance ({attendance_pct}%) is within the 10% condonation window under Clause 10.1. "
            f"Requires Principal's sanction upon submission of medical records or institutional representation proof."
        )
    else:
        eligible = False
        status = "INELIGIBLE"
        condonation_required = False
        attendance_deficit = round(75.0 - attendance_pct, 1)
        applicable_clause = "Clause 10.3 & Addendum Clause 21.2 (Attendance Shortage — 'DX' Grade)"
        recommendation = (
            f"Attendance ({attendance_pct}%) is below the absolute 75.0% condonation floor. "
            f"Bars student from SEE, resulting in a 'DX' grade. Student must re-register in subsequent semesters "
            f"or clear through the 10-week Summer Semester under Addendum Clause 21.3."
        )

    # Check backlogs against Vertical Progression (Clause 13.2.1: max 16 credits of F grades)
    backlog_warning = None
    if backlogs > 0:
        if backlog_credits <= 16:
            backlog_warning = (
                f"Student has {backlogs} backlog course(s) ({backlog_credits} credits). "
                f"Within the 16-credit vertical progression ceiling for Second Year under Clause 13.2.1."
            )
        else:
            backlog_warning = (
                f"⚠️ Backlog credits ({backlog_credits}) exceed the 16-credit progression limit. "
                f"Review vertical progression requirements under Clause 13.2.1 & Addendum Clause 21.5."
            )

    return {
        "usn": student.get("usn"),
        "name": student.get("name"),
        "department": student.get("department"),
        "semester": student.get("semester"),
        "attendance_pct": attendance_pct,
        "cgpa": cgpa,
        "eligible": eligible,
        "status": status,
        "applicable_clause": applicable_clause,
        "leave_type": leave_type,
        "backlogs": backlogs,
        "backlog_credits": backlog_credits,
        "backlog_warning": backlog_warning,
        "condonation_required": condonation_required,
        "attendance_deficit": attendance_deficit,
        "proctor": student.get("proctor"),
        "hod": student.get("hod"),
        "recommendation": recommendation,
    }


@mcp.tool()
def draft_official_petition(
    usn: str,
    student_name: str,
    clause_id: str,
    reason: str,
    event_date: str = "Upcoming Examination Cycle",
    event_name: str = "Academic / Extracurricular Representation",
    course_code: str = "ALL"
) -> dict:
    """
    Generate a formal Principal / HOD petition letter citing official MSRIT
    Autonomous Regulations clauses.

    Args:
        usn: Student USN (e.g. "1MS22CS045")
        student_name: Full name of student
        clause_id: Policy clause being invoked (e.g. "Clause 10.1", "Clause 8.10")
        reason: Reason for petition
        event_date: Event / medical date
        event_name: Event / exam description
        course_code: Course code(s) involved

    Returns:
        Formatted official petition letter text and metadata.
    """
    department = "Computer Science & Engineering"
    semester = 5
    hod_name = "The Head of Department"
    proctor = "Designated Proctor"

    try:
        students = load_students_db()
        usn_upper = usn.strip().upper()
        student_list = list(students.values()) if isinstance(students, dict) else students
        for s in student_list:
            if s.get("usn", "").upper() == usn_upper:
                department = s.get("department", department)
                semester = s.get("semester", semester)
                hod_name = s.get("hod", hod_name)
                proctor = s.get("proctor", proctor)
                if not student_name or student_name == "Student":
                    student_name = s.get("name", student_name)
                break
    except Exception:
        pass

    current_date = datetime.date.today().strftime("%d %B %Y")
    clause_lower = clause_id.lower()

    # Determine petition body based on cited clause
    if "10.1" in clause_lower or "attendance" in reason.lower() or "condonation" in reason.lower():
        sub_line = f"Application for Attendance Condonation under Clause 10.1 ({course_code})"
        addressed_to = f"The Principal\n(Through the Head of Department, {department})\nRamaiah Institute of Technology\nBangalore - 560054"
        body_para = (
            f"I am writing to respectfully petition for attendance condonation in accordance with "
            f"Clause 10.1 of the MSRIT Autonomous Degree Regulations. Clause 10.1 provides for "
            f"condonation of up to 10% of attendance by the Principal for valid reasons including "
            f"medical emergencies, participation in sports, cultural activities, seminars, workshops, "
            f"and paper presentations at University, State, or National levels.\n\n"
            f"Due to {reason} on/during {event_date}, my attendance in {course_code} currently stands "
            f"within the permitted condonation window (75.0% – 84.99%). I have maintained satisfactory "
            f"academic performance and active engagement throughout the semester."
        )
        enclosures = [
            "Medical Certificate / Hospital Discharge Summary (if medical)",
            "Official Event Participation Certificate / Paper Acceptance Letter",
            "Attendance Ledger verified by Proctor / Course Instructor",
            "Proctor Recommendation Slip",
        ]
        undertaking = (
            "I hereby undertake to attend all remaining lectures, tutorials, and practical sessions "
            "without further absence, complete all pending continuous assessments, and abide by the "
            "final decision of the Principal and Examination Section."
        )

    elif "8.10" in clause_lower or "make-up" in reason.lower() or "make up" in reason.lower() or "8.11" in clause_lower:
        sub_line = f"Petition for Award of Transitional Grade 'I' / 'X' and Make-Up SEE under Clause 8.10 / 8.11"
        addressed_to = f"The Controller of Examinations\n(Through HOD, {department})\nRamaiah Institute of Technology\nBangalore - 560054"
        body_para = (
            f"I respectfully submit this petition requesting permission to appear for the Make-Up "
            f"Semester End Examination (SEE) in {course_code} as per Clause 8.10 and Clause 8.11 "
            f"of the MSRIT Autonomous Regulations.\n\n"
            f"Under Clause 8.10, a student possessing satisfactory attendance and passing CIE marks "
            f"who was unavoidably absent from SEE due to verified medical accident or serious illness "
            f"is eligible for award of Transitional Grade 'I' and subsequent appearance in the Make-Up SEE.\n\n"
            f"Reason for absence: {reason} on {event_date}."
        )
        enclosures = [
            "Official Doctor's Certificate from a Registered Medical Practitioner",
            "Hospitalisation Records / Emergency Room Admission Record",
            "CIE Marks Card demonstrating passing CIE marks (>= 40%)",
            "Class Proctor Endorsement Letter",
        ]
        undertaking = (
            "I confirm that my CIE marks are intact as per Clause 9.6, and I undertake to register "
            "for the Make-Up Examination schedule as declared by the Examination Section."
        )

    elif "21.3" in clause_lower or "summer" in reason.lower():
        sub_line = f"Application for Summer Semester Course Registration under Addendum Clause 21.3"
        addressed_to = f"The Dean (Academics)\n(Through HOD, {department})\nRamaiah Institute of Technology\nBangalore - 560054"
        body_para = (
            f"I respectfully submit this application to register for the 10-week Summer Semester "
            f"as sanctioned under Addendum Clause 21.3 (VTU Notification No. VTU/PS/2024-25/223) "
            f"to clear the course: {course_code} (Reason: {reason}).\n\n"
            f"As stipulated in Addendum Clause 21.3, the Summer Semester is specifically open for "
            f"clearing 'DX'/'NP' grade courses and backlog 'F'/'AB' courses, with maximum permissible "
            f"registration of up to 14 credits for UG Engineering programmes."
        )
        enclosures = [
            "Semester Grade Card showing 'DX' or 'F' status",
            "Course Registration Slip verified by Proctor",
            "Prescribed Summer Semester Fee Payment Receipt",
        ]
        undertaking = (
            "I undertake to satisfy the mandatory 8-week instruction attendance requirement and appear "
            "for the final examinations as scheduled, noting that no make-up exams exist for the Summer Semester."
        )

    else:
        sub_line = f"Academic Representation and Accommodation Petition under {clause_id}"
        addressed_to = f"The Head of Department\nDepartment of {department}\nRamaiah Institute of Technology\nBangalore - 560054"
        body_para = (
            f"I am writing to formally submit an academic representation under {clause_id} of the "
            f"MSRIT Autonomous Regulations. Reason: {reason} on {event_date}."
        )
        enclosures = ["Supporting Documents and Verification Proof", "Proctor Endorsement"]
        undertaking = "I undertake to adhere strictly to all academic regulations of Ramaiah Institute of Technology."

    petition_text = f"""DATE: {current_date}

TO:
{addressed_to}

FROM:
Student Name : {student_name}
USN          : {usn}
Department   : {department}
Semester     : {semester}
Proctor      : {proctor}

SUBJECT: {sub_line}

Respected Sir/Madam,

{body_para}

{undertaking}

Thanking you.

Yours faithfully,

_______________________
{student_name}
USN: {usn}
Department of {department}
Ramaiah Institute of Technology, Bangalore

ENCLOSURES:
""" + "\n".join([f"  {idx}. {enc}" for idx, enc in enumerate(enclosures, 1)]) + f"""

--------------------------------------------------------------------------------
RECOMMENDATION OF PROCTOR / HOD:
[  ] Recommended & Forwarded to Principal for Condonation / Examination Clearance
[  ] Additional Documentary Verification Requested
Remarks: ______________________________________________________________________

Signature of Proctor: ____________________    Signature of HOD: ____________________
Date: {current_date}
"""

    filename = f"MSRIT_Petition_{usn}_{clause_id.replace(' ', '_').replace('.', '_')}_{datetime.date.today()}.txt"

    return {
        "petition_text": petition_text.strip(),
        "filename": filename,
        "clause_cited": clause_id,
        "student_usn": usn,
        "student_name": student_name,
        "department": department,
        "generated_at": datetime.datetime.now().isoformat(),
        "word_count": len(petition_text.split()),
    }


if __name__ == "__main__":
    mcp.run()
