import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, Modal, TouchableOpacity, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import axios from 'axios';
import AntDesign from '@expo/vector-icons/AntDesign'; // Import AntDesign icon


const App = () => {
  const [chapter, setChapter] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [messages, setMessages] = useState<{ type: string; text: string }[]>([]);

  const backendURL = 'http://192.168.2.24:5000/chat'; // Replace with your backend URL

  const handleSubmit = async () => {
    if (!chapter || !question) {
      alert('Please select a chapter and ask a question.');
      return;
    }

    setLoading(true);

    setMessages((prevMessages) => [
      ...prevMessages,
      { type: 'user', text: question }
    ]);

    setQuestion(''); 

    try {
      const response = await axios.post(backendURL, {
        chapter,
        question,
      });

      const answerResponse = response.data.answer;

      setMessages((prevMessages) => [
        ...prevMessages,
        { type: 'bot', text: answerResponse }
      ]);

      setAnswer(answerResponse); 
    } catch (error) {
      console.error(error);
      alert('Error while fetching answer');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Professor Lambard</Text>

      {/* Chapter Picker Button */}
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Select Chapter:</Text>
        <TouchableOpacity 
          style={styles.pickerButton}
          onPress={() => setModalVisible(true)}
        >
          <Text style={styles.pickerButtonText}>
            {chapter ? `Chapter ${chapter}` : 'Select Chapter'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Modal for Chapter Picker */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalBackground}>
          <View style={styles.modalContainer}>
            <Text style={styles.modalTitle}>Select a Chapter</Text>
            <TouchableOpacity onPress={() => { setChapter('fluid-mechanics'); setModalVisible(false); }}>
              <Text style={styles.modalItem}>Chapter 9</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => { setChapter('thermodynamics'); setModalVisible(false); }}>
              <Text style={styles.modalItem}>Chapter 11</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => { setChapter('gravitation'); setModalVisible(false); }}>
              <Text style={styles.modalItem}>Chapter 7</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setModalVisible(false)}>
              <Text style={styles.modalClose}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Display Conversation */}
      <View style={styles.conversationContainer}>
        {messages.map((msg, index) => (
          <View key={index} style={msg.type === 'user' ? styles.userMessage : styles.botMessage}>
            <Text style={styles.messageText}>{msg.text}</Text>
          </View>
        ))}
      </View>

      {/* Question Input with "Go" arrow */}
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Ask a Question:</Text>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.input}
            value={question}
            onChangeText={setQuestion}
            placeholder="Type your question"
          />
          <TouchableOpacity onPress={handleSubmit} style={styles.goButton}>
            <AntDesign name="rightcircle" size={24} color="black" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Loading Text */}
      {loading && <Text style={styles.loadingText}>Loading...</Text>}
    </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
  },
  inputContainer: {
    marginBottom: 20,
    paddingHorizontal: 10,
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  input: {
    height: 40,
    flex: 1,
    borderColor: '#ccc',
    borderWidth: 1,
    paddingLeft: 10,
    borderRadius: 5,
  },
  goButton: {
    marginLeft: 10,
    padding: 5,
    justifyContent: 'center',
    alignItems: 'center',
  },
  pickerButton: {
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    backgroundColor: '#f7f7f7',
    paddingHorizontal: 10,
  },
  pickerButtonText: {
    fontSize: 16,
    color: '#333',
  },
  modalBackground: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContainer: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 10,
    width: '80%',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  modalItem: {
    fontSize: 16,
    paddingVertical: 10,
    textAlign: 'center',
    color: '#007bff',
  },
  modalClose: {
    fontSize: 16,
    paddingVertical: 10,
    textAlign: 'center',
    color: '#ff0000',
  },
  conversationContainer: {
    marginBottom: 20,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#cfe2f3',
    padding: 10,
    borderRadius: 10,
    marginVertical: 5,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#f1f1f1',
    padding: 10,
    borderRadius: 10,
    marginVertical: 5,
  },
  messageText: {
    fontSize: 16,
    color: '#333',
  },
  loadingText: {
    textAlign: 'center',
    fontSize: 18,
    color: '#888',
    marginTop: 10,
  },
});

export default App;
