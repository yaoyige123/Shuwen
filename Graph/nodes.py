import time
import json
from typing import Callable

from app.graphqa import resolve_for_risk
from Chains.graph_qa_chain import get_graph_qa_chain
from Graph.state import MedicalState, InputState
from Chains.assistant2agent_chain import get_primary_assistant_chain
from Chains.information_chain import get_information_chain
from langchain_core.messages import ToolMessage,HumanMessage,AIMessage
from Chains.extract_info_chain import get_extract_info_chain
from Chains.history_chain import get_history_chain
from Chains.risk_chain import get_risk_chain
from typing import List, Dict, Any
from Chains.analgesia_chain import get_analgesia_chain


def get_tool_calls(message):
    """Read tool calls produced by either old or new LangChain versions."""
    calls = getattr(message, "tool_calls", None)
    if calls:
        return calls
    normalized = []
    for call in getattr(message, "additional_kwargs", {}).get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        normalized.append({
            "id": call.get("id", ""),
            "name": function.get("name") or call.get("name"),
            "args": arguments,
        })
    return normalized

def Primary_Assistant(state: MedicalState):
    try:
        primary_chain = get_primary_assistant_chain()
        messages = state.get('messages', [])
        user_information = state.get('user_information', '')
        result = primary_chain.invoke({
            "messages": messages,
            "user_information": user_information
        })
        
        # 确保返回完整的状态
        return {
            "messages": result,
        }
    except Exception as e:
        print(f"主助手处理时出错: {str(e)}")
        # 返回默认状态
        return {
            "messages": [],
        }


#用于从Primary_Assistant分支到Agent,同时跟踪dialog_state

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: MedicalState) -> dict:
        # 获取最后一条消息
        last_message = state["messages"][-1]
        tool_messages = []
        
        # 检查是否有工具调用
        tool_calls = get_tool_calls(last_message)
        if tool_calls:
            for tool_call in tool_calls:
                # 处理parallel工具调用
                if tool_call.get("name") == "multi_tool_use.parallel":
                    tool_uses = tool_call.get("args", {}).get("tool_uses", [])
                    for tool_use in tool_uses:
                        tool_messages.append(
                            ToolMessage(
                                content=f'''记住，你现在是{assistant_name}，你的任务尚未完成。反思上述主任医生和患者之间的对话。\n如果患者改变主意或需要帮助处理其他任务，请调用CompleteOrEscalate函数让主要主任医师接管。不提及你是谁——只需作为助理行事。''',
                                tool_call_id=tool_call["id"]
                            )
                        )
                # 处理普通工具调用
                else:
                    agent_id = tool_call.get("args", {}).get("agent_id", []) 
                    tool_messages.append(
                        ToolMessage(
                            content=f'''记住，你现在是{assistant_name}，你的任务尚未完成。反思上述主任医生和患者之间的对话。\n如果患者改变主意或需要帮助处理其他任务，请调用CompleteOrEscalate函数让主要主任医师接管。不提及你是谁——只需作为助理行事。''',
                            tool_call_id=tool_call["id"],
                            args={"agent_id": agent_id}
                        )
                    )
        
        if not tool_messages:
            print("警告：未找到工具调用")
            return {"dialog_state": new_dialog_state}
        if not agent_id:
            return {
                "messages": tool_messages,
                "dialog_state": new_dialog_state,
            }
        else:
            return {
                "messages": tool_messages,
                "dialog_state": new_dialog_state,
                "agent_id": agent_id
            }
    return entry_node

Entry_Information_Agent = create_entry_node(assistant_name="确认患者身份信息的医疗助手", 
                                            new_dialog_state="verify_information")

def Information_Agent(state: MedicalState):
    try:
        information_chain = get_information_chain()
        messages = state.get('messages', [])
        user_information = state.get('user_information', '')
        
        result = information_chain.invoke({
            "messages": messages,
            "user_information": user_information,
        })
        # print("AI 返回内容：", repr(result.content))
        # print("附加信息：", result.additional_kwargs)
        return {
            "messages": result,
        }
    except Exception as e:
        print(f"信息确认助手处理时出错: {str(e)}")
        raise

def Extract_Info_Agent(state: MedicalState):
    try:
        extract_info_chain = get_extract_info_chain()
        
        # 获取所有消息
        messages = state.get('messages', [])
        
        # 找到最后一条human message的位置
        last_human_idx = None
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        
        if last_human_idx is None:
            # 如果没有找到human message，返回空状态
            return {
                "user_information": "",
                "medical_history": [],
                "medicine_taking": [],
            }
        
        # 获取最后一条human message和它的上一条消息（如果存在）
        relevant_messages = []
        if last_human_idx > 0:
            relevant_messages.append(messages[last_human_idx - 1])
        relevant_messages.append(messages[last_human_idx])
        
        # 获取当前状态，如果不存在则使用默认值
        user_information = state.get('user_information', '')
        
        # 消息转换为文本格式
        messages_text = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in relevant_messages])
        
        result =extract_info_chain.invoke({
            "user_input": messages_text,
            "user_information": user_information,
        })
        
        # 更新user_information
        updated_user_information = f"""
        姓名: {result.patient}
        年龄: {result.age}
        手术: {result.surgery}
        麻醉方式: {result.anesthesia}
        其他信息: {result.additional_info}
        """
        
        # 返回完整状态
        return {
            "user_information": updated_user_information.strip(),
            "medical_history": list(result.medical_history) if result.medical_history else [],
            "medicine_taking": list(result.medicine_taking) if result.medicine_taking else [],
        }
    except Exception as e:
        print(f"提取信息时出错: {str(e)}")
        raise

# 创建Entry_History_Agent
Entry_History_Agent = create_entry_node(
    assistant_name="采集病史的麻醉医生",
    new_dialog_state="history_taking"
)

def History_Agent(state: MedicalState):
    try:
        agent_id = state.get('agent_id', '')
        messages = state.get('messages', [])

        # On entry, ask the first question explicitly. If the full information
        # verification transcript is sent straight to the model, it can mistake
        # that completed phase for a completed history assessment and immediately
        # call CompleteOrEscalate, producing no visible reply.
        if messages and isinstance(messages[-1], ToolMessage):
            # 每个子助手的第一句话：把同类问题合并成一条，减少轮次
            opening_questions = {
                1: "您以前做过手术或接受过麻醉吗？当时用的是哪种麻醉？术后有没有恶心呕吐？平时坐车坐船会晕吗？",
                2: "想先了解一下您的慢性病情况：有没有高血压、冠心病、心律失常？有没有哮喘、慢阻肺、慢性咳嗽咳痰？有没有脑梗、癫痫、头晕头痛这些？",
                3: "最近睡眠和胃口怎么样？大小便正常吗？抽烟喝酒吗？目前在吃什么药或者保健品？",
            }
            return {
                "messages": AIMessage(
                    content=opening_questions.get(
                        agent_id, "请您介绍一下目前的身体情况。"
                    )
                )
            }

        history_chain = get_history_chain(agent_id)
        user_information = state.get('user_information', '')
        medical_history = state.get('medical_history', [])
        medicine_taking = state.get('medicine_taking', [])
        result = history_chain.invoke({
            "messages": messages,
            "user_information": user_information,
            "medical_history": medical_history,
            "medicine_taking": medicine_taking,
        })     
        return {
            "messages": result,
        }
    except Exception as e:
        print(f"病史采集助手处理时出错: {str(e)}")
        return {
            "messages": [],
        }

# 创建Entry_Risk_Agent
Entry_Risk_Agent = create_entry_node(
    assistant_name="进行麻醉风险评估的麻醉医生",
    new_dialog_state="risk_assessment"
)

def Risk_Agent(state: MedicalState):
    try:
        risk_count = state.get('risk_count') or 0
        risk_chain = get_risk_chain(risk_count)
        messages = state.get('messages', [])
        user_information = state.get('user_information', '')
        medical_history = state.get('medical_history', [])
        medicine_taking = state.get('medicine_taking', [])
        if risk_count > 0:
            # 计算要删除的消息的位置（从末尾数起的第几条）
            target_index = -(risk_count * 2)
            # 保留目标消息之前和之后的所有消息，只删除目标位置的消息
            messages = messages[:target_index] + messages[target_index + 1:]
        
        # 获取会话ID
        session_id = state.get('session_id', 'default')

        # 分层取数（见 app/graphqa.py）：
        #   最多等一次预取（短超时）→ global → local → 词面兜底 → 档案+通用清单。
        # 拿不到图谱结果时不再写死一句"未完成"，而是走带局限性说明的降级文案。
        graph_qa_result, retrieval_path = resolve_for_risk(session_id, state)
        print(f"会话 {session_id}: 风险告知依据来源 = {retrieval_path}")
        
        # 调用风险评估链
        result = risk_chain.invoke({
                "messages": messages,
                "user_information": user_information,
                "medical_history": medical_history,
                "medicine_taking": medicine_taking,
                "graph_qa_result": graph_qa_result,
                "risk_count": risk_count
        })
        return {
            "messages": result,
            "risk_count": risk_count+1
        }
    except Exception as e:
        print(f"风险评估助手处理时出错: {str(e)}")
        raise
    
def Analgesia_Agent(state: MedicalState):
    """处理镇痛相关的对话"""
    try:
        # 获取chain
        analgesia_chain = get_analgesia_chain()
        
        # 获取当前消息
        messages = state.get('messages', [])
        
        # 准备上下文信息
        context = {
            "messages": messages,  # 只获取最新消息
            "user_information": state.get("user_information", ""),
            "medical_history": state.get("medical_history", ""),
            "medicine_taking": state.get("medicine_taking", "")
        }
        
        # 调用chain处理消息
        response = analgesia_chain.invoke(context)

        return {
            "messages":response
        }
    except Exception as e:
        print(f"镇痛助手处理时出错: {str(e)}")
        return {
            "messages": [],
        }

def Graph_QA_Agent(state: MedicalState):
    """
    使用知识图谱分析患者风险的Agent。
    返回风险分析结果和更新后的状态。
    """
    try:
        # 获取chain
        graph_qa_chain = get_graph_qa_chain()
        
        # 准备输入
        messages = state.get('messages', [])
        user_information = state.get('user_information', '')
        medical_history = state.get('medical_history', [])
        medicine_taking = state.get('medicine_taking', [])
        
        # 调用chain
        result = graph_qa_chain.invoke({
            "messages": messages,
            "user_information": user_information,
            "medical_history": medical_history,
            "medicine_taking": medicine_taking,
        })
        
        # 处理返回结果
        risk_analysis = result.get('risk_analysis', '')
        
        # 创建AI消息
        ai_message = AIMessage(
            content=f"""根据知识图谱分析，患者的风险评估如下：

        {risk_analysis}

        请根据这些信息为基础，与患者进行沟通交流，告知麻醉手术风向。
        """
        )
        
        # 返回更新后的状态
        return {
            "messages": messages + [ai_message],
            "graph_qa_result": risk_analysis
        }
        
    except Exception as e:
        print(f"图谱QA助手处理时出错: {str(e)}")
        # 发生错误时返回原始状态
        return {
            "messages": messages,
        }

# 创建Entry_Analgesia_Agent
Entry_Analgesia_Agent = create_entry_node(
    assistant_name="介绍镇痛方案的麻醉医生",
    new_dialog_state="analgesia"
)

def leave_history_agent(state: MedicalState):
    return {
        "extract_info_result": "leave_history_agent"
    }


    


    


    


    
