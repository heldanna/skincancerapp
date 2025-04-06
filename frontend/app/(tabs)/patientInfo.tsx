/*import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { useNavigation } from 'expo-router';

export default function PatientInfo() {
  const navigation = useNavigation();
  const [name, setName] = useState('');
  const [dob, setDob] = useState('');
  const [room, setRoom] = useState('');

  return (
    <View style={styles.container}>
      <Image source={require('@/assets/images/MoffittLogo.png')} style={styles.logo} />

      <View style={styles.content}>
        <TouchableOpacity
          style={styles.scanButton}
          onPress={() => navigation.navigate('BarcodeScan')}
        >
          <Text style={styles.buttonText}>Scan Patient Barcode</Text>
        </TouchableOpacity>

        <Text style={styles.orText}>—  OR  —</Text>

        <TextInput
          placeholder="Name"
          value={name}
          onChangeText={setName}
          style={styles.input}
        />
        <TextInput
          placeholder="DOB"
          value={dob}
          onChangeText={setDob}
          style={styles.input}
        />
        <TextInput
          placeholder="Room #"
          value={room}
          onChangeText={setRoom}
          style={styles.input}
        />

        <TouchableOpacity
          style={styles.confirmButton}
          onPress={() => navigation.navigate('SkinPic')}
        >
          <Text style={styles.buttonText}>Confirm</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  logo: {
    width: 50,
    height: 50,
    resizeMode: 'contain',
    position: 'absolute',
    top: 40,
    left: 20,
  },
  content: {
    flex: 1,
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 80,
    marginBottom: 30,
  },
  scanButton: {
    backgroundColor: '#035C96',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
  },
  confirmButton: {
    backgroundColor: '#035C96',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
  },
  orText: {
    marginVertical: 20,
    fontSize: 16,
    color: '#035C96',
    fontWeight: '600',
  },
  input: {
    width: '100%',
    backgroundColor: '#F3F3F7',
    padding: 12,
    borderRadius: 10,
    marginBottom: 10,
    fontSize: 16,
  },
});

import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { useNavigation } from 'expo-router';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';

// Define navigation types
type RootStackParamList = {
  skinPic: undefined;
};

// Set up navigation type
type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'skinPic'>;

export default function PatientInfo() {
  const navigation = useNavigation<NavigationProp>();
  const [name, setName] = useState<string>('');
  const [dob, setDob] = useState<string>('');
  const [room, setRoom] = useState<string>('');

  return (
    <View style={styles.container}>
      <Image source={require('@/assets/images/MoffittLogo.png')} style={styles.logo} />

      <View style={styles.content}>
        <TextInput
          placeholder="Name"
          value={name}
          onChangeText={setName}
          style={styles.input}
        />
        <TextInput
          placeholder="DOB"
          value={dob}
          onChangeText={setDob}
          style={styles.input}
        />
        <TextInput
          placeholder="Room #"
          value={room}
          onChangeText={setRoom}
          style={styles.input}
        />

        <TouchableOpacity
          style={styles.confirmButton}
          onPress={() => navigation.navigate('skinPic')}
        >
          <Text style={styles.buttonText}>Confirm</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  logo: {
    width: 50,
    height: 50,
    resizeMode: 'contain',
    position: 'absolute',
    top: 40,
    left: 20,
  },
  content: {
    flex: 1,
    justifyContent: 'center', // Center vertically
    alignItems: 'center', // Center horizontally
    marginTop: 80,
  },
  confirmButton: {
    backgroundColor: '#035C96',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
    marginTop: 20, // Space after last input
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
  },
  input: {
    width: '80%', // Adjusted width for better centering
    backgroundColor: '#F3F3F7',
    padding: 12,
    borderRadius: 10,
    marginBottom: 15,
    fontSize: 16,
  },
});

import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Image, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';

export default function PatientInfo() {
  const router = useRouter();
  const [name, setName] = useState<string>('');
  const [dob, setDob] = useState<string>('');
  const [room, setRoom] = useState<string>('');

  return (
    <View style={styles.container}>
      <Image
        source={require('@/assets/images/MoffittCancerCenterLogo.jpg')}
        style={styles.logo}
        resizeMode="contain"
      />
      <Text style={styles.title}>Patient Information</Text>

      <View style={styles.formContainer}>
        <TextInput
          placeholder="Name"
          value={name}
          onChangeText={setName}
          style={styles.input}
        />
        <TextInput
          placeholder="DOB"
          value={dob}
          onChangeText={setDob}
          style={styles.input}
        />
        <TextInput
          placeholder="Room #"
          value={room}
          onChangeText={setRoom}
          style={styles.input}
        />

        <TouchableOpacity style={styles.confirmButton} onPress={() => router.push('/(tabs)/skinPic')}>
          <Text style={styles.buttonText}>Confirm</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 80,
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  logo: {
    width: 250,
    height: 80,
    marginBottom: 30,
  },
  title: {
    fontSize: 26,
    textAlign: 'center',
    fontWeight: '600',
    color: '#035C96',
    marginBottom: 30,
  },
  formContainer: {
    width: '80%',
    alignItems: 'center',
  },
  input: {
    width: '100%',
    backgroundColor: '#F3F3F7',
    padding: 12,
    borderRadius: 10,
    marginBottom: 15,
    fontSize: 16,
  },
  confirmButton: {
    backgroundColor: '#035C96',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 8,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});*/
// app/(tabs)/PatientInfo.tsx

import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';

export default function PatientInfo() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [dob, setDob] = useState('');
  const [room, setRoom] = useState('');

  const handleConfirm = () => {
    router.push('/skinPic');
  };

  return (
    <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      <Image source={require('@/assets/images/MoffittLogo.png')} style={styles.logo} />
      <Text style={styles.header}>Enter Patient Information</Text>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          placeholder="Patient Name"
          value={name}
          onChangeText={setName}
        />
        <TextInput
          style={styles.input}
          placeholder="Date of Birth"
          value={dob}
          onChangeText={setDob}
        />
        <TextInput
          style={styles.input}
          placeholder="Room Number"
          value={room}
          onChangeText={setRoom}
        />
      </View>

      <TouchableOpacity style={styles.button} onPress={handleConfirm}>
        <Text style={styles.buttonText}>Confirm</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#fff',
  },
  logo: {
    position: 'absolute',
    top: 40,
    right: 20,
    width: 60,
    height: 60,
    resizeMode: 'contain',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 40,
    textAlign: 'center',
    color: '#035C96',
  },
  inputContainer: {
    width: '100%',
    alignItems: 'center',
    marginBottom: 20,
  },
  input: {
    width: '100%',
    height: 50,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 15,
    marginVertical: 10,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#035C96',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 10,
    marginTop: 20,
    width: '100%',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
    fontWeight: 'bold',
  },
});
