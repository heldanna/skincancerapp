import React, { useState } from 'react';
import { View, Text, Button, Image, StyleSheet, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';

const SkinPic = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const navigation = useNavigation();

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Permission to access media library is required!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
    }
  };

  const handleAnalysis = () => {
    navigation.navigate('Results');
  };

  return (
    <SafeAreaView style={styles.container}>
      <Image source={require('@/assets/images/MoffittLogo.png')} style={styles.logo} />
      <Text style={styles.title}>Upload Skin Lesion Image</Text>

      <TouchableOpacity style={styles.uploadButton} onPress={pickImage}>
        <Text style={styles.buttonText}>Choose from Gallery</Text>
      </TouchableOpacity>

      {imageUri && <Image source={{ uri: imageUri }} style={styles.preview} />}

      {imageUri && (
        <TouchableOpacity style={styles.analyzeButton} onPress={handleAnalysis}>
          <Text style={styles.buttonText}>Start Analysis</Text>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
};

export default SkinPic;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    justifyContent: 'flex-start',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  logo: {
    width: 100,
    height: 50,
    resizeMode: 'contain',
    alignSelf: 'flex-end',
    marginTop: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginVertical: 20,
  },
  uploadButton: {
    backgroundColor: '#035C96',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    marginBottom: 20,
  },
  analyzeButton: {
    backgroundColor: '#035C96',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 16,
  },
  preview: {
    width: 250,
    height: 250,
    borderRadius: 10,
    resizeMode: 'cover',
    marginTop: 20,
  },
});
